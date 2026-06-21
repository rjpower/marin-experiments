"""Isolated command execution for agent tool use, via gVisor (runsc).

The OpenThoughts terminal agent issues shell commands that we must NOT run on the
training host. Terminal-Bench tasks ship their own Docker environment, so the
sandbox model is: run each task's image as a container under the **runsc** OCI
runtime (gVisor), and ``docker exec`` the agent's commands into it.

This runs inside the iris TPU task container, which is ``--privileged`` (iris adds
that for accelerators), so rootful gVisor + a task-local dockerd work. The custom
task image (`docker/Dockerfile.agent-task`) ships `runsc`, `docker`, and a
`/etc/docker/daemon.json` registering the `runsc` runtime.

Implementations:
  * :class:`GvisorContainerSandbox` -- a long-lived container per task, started
    with ``docker run --runtime=runsc``; the production path.
  * :class:`LocalUnsafeSandbox` -- plain subprocess, NO isolation. For developing
    the agent-loop / grading logic on a laptop only; never use on untrusted code.

Use :func:`make_sandbox` to pick by ``OTA_SANDBOX`` env (``gvisor`` | ``local``).
"""

import dataclasses
import os
import shutil
import subprocess
import time
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class ExecResult:
  """Result of one command execution."""

  stdout: str
  stderr: str
  exit_code: int
  timed_out: bool = False


class Sandbox(Protocol):
  """A place to run agent shell commands in isolation."""

  def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
    """Runs ``command`` (a shell string) and returns its result."""
    ...

  def close(self) -> None:
    """Tears down any container / resources."""
    ...


def _run(argv: list[str], *, timeout: float, input_text: str | None = None) -> ExecResult:
  """Runs a subprocess with a hard timeout, capturing stdout/stderr."""
  try:
    proc = subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return ExecResult(proc.stdout, proc.stderr, proc.returncode)
  except subprocess.TimeoutExpired as e:
    return ExecResult(e.stdout or "", (e.stderr or "") + "\n[timeout]", 124, timed_out=True)


# dockerd flags for running nested inside a container: vfs avoids nested-overlayfs
# failures, and we disable the bridge/iptables since sandbox containers use
# ``--network none`` (no docker bridge needed). Override via ``DOCKERD_ARGS``.
_DEFAULT_DOCKERD_ARGS = ["--storage-driver=vfs", "--iptables=false", "--bridge=none"]
_DOCKERD_LOG = "/tmp/dockerd.out"


def ensure_dockerd(*, timeout: float = 120.0) -> None:
  """Starts a task-local dockerd if the socket isn't already up, and waits for it.

  Idempotent. Requires the ``--privileged`` task container (TPU jobs have it).
  Runs dockerd with vfs storage + no bridge/iptables, which is what lets it come
  up nested inside the iris task container.

  Raises:
    RuntimeError: if dockerd does not become ready within ``timeout`` (the error
      includes the tail of dockerd's own log to make the cause visible).
  """
  if shutil.which("docker") is None:
    raise RuntimeError("docker not found; use the openthoughts-agent-task image.")
  if _run(["docker", "info"], timeout=10).exit_code == 0:
    return
  extra = os.environ.get("DOCKERD_ARGS")
  args = extra.split() if extra else _DEFAULT_DOCKERD_ARGS
  subprocess.Popen(
      ["dockerd", *args],
      stdout=open(_DOCKERD_LOG, "w"),
      stderr=subprocess.STDOUT,
  )
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if _run(["docker", "info"], timeout=10).exit_code == 0:
      return
    time.sleep(1.0)
  tail = ""
  try:
    with open(_DOCKERD_LOG) as f:
      tail = "".join(f.readlines()[-25:])
  except OSError:
    pass
  raise RuntimeError(f"dockerd not ready after {timeout}s. dockerd log tail:\n{tail}")


class GvisorContainerSandbox:
  """A Terminal-Bench task environment running under gVisor.

  Starts ``image`` as a detached container with the ``runsc`` runtime and execs
  agent commands into it via ``docker exec``. The container is removed on
  :meth:`close`.
  """

  def __init__(
      self,
      image: str,
      *,
      workdir: str | None = None,
      runtime: str = "runsc",
      network: str = "none",
      name: str | None = None,
      mem_limit: str = "4g",
      cpus: str = "2",
  ):
    ensure_dockerd()
    self.image = image
    self.workdir = workdir  # None => use the image's own WORKDIR
    self._name = name or f"ota-task-{os.getpid()}-{int(time.monotonic()*1000)}"
    argv = [
        "docker", "run", "-d", "--rm",
        "--runtime", runtime,
        "--network", network,
        "--memory", mem_limit,
        "--cpus", cpus,
    ]
    if workdir:
      argv += ["--workdir", workdir]
    # Keep the container alive with a no-op PID 1 so we can exec into it.
    argv += ["--name", self._name, image, "sleep", "infinity"]
    res = _run(argv, timeout=300)
    if res.exit_code != 0:
      raise RuntimeError(
          f"failed to start sandbox container from {image!r}: {res.stderr}"
      )

  @property
  def name(self) -> str:
    """The container name (for ``docker cp`` / grading)."""
    return self._name

  def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
    argv = ["docker", "exec"]
    if self.workdir:
      argv += ["--workdir", self.workdir]
    argv += [self._name, "bash", "-lc", command]
    return _run(argv, timeout=timeout)

  def copy_in(self, local_path: str, container_path: str, *, timeout: float = 120.0) -> ExecResult:
    """Copies a host path into the container (``docker cp``)."""
    return _run(["docker", "cp", local_path, f"{self._name}:{container_path}"], timeout=timeout)

  def close(self) -> None:
    _run(["docker", "rm", "-f", self._name], timeout=30)


def build_image(context_dir: str, tag: str, *, timeout: float = 1200.0) -> ExecResult:
  """Builds a Docker image from ``context_dir`` (must contain a Dockerfile)."""
  ensure_dockerd()
  return _run(["docker", "build", "-t", tag, context_dir], timeout=timeout)


class LocalUnsafeSandbox:
  """Plain subprocess execution with NO isolation. Dev/testing only.

  Runs commands directly in a temp dir on the host. Use ONLY to exercise the
  agent-loop / grading logic with trusted commands; never on model-generated code.
  """

  def __init__(self, workdir: str | None = None):
    import tempfile

    self.workdir = workdir or tempfile.mkdtemp(prefix="ota-local-")

  def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
    return _run(["bash", "-lc", command], timeout=timeout, input_text=None)

  def close(self) -> None:
    pass


def make_sandbox(image: str | None = None, **kwargs) -> Sandbox:
  """Returns a sandbox per the ``OTA_SANDBOX`` env (default ``gvisor``).

  Args:
    image: the Docker image for the gvisor sandbox (required for ``gvisor``).
    **kwargs: forwarded to the sandbox constructor.

  Raises:
    ValueError: for an unknown ``OTA_SANDBOX`` or a missing image.
  """
  kind = os.environ.get("OTA_SANDBOX", "gvisor").lower()
  if kind == "gvisor":
    if not image:
      raise ValueError("gvisor sandbox requires an image.")
    return GvisorContainerSandbox(image, **kwargs)
  if kind == "local":
    return LocalUnsafeSandbox(**{k: v for k, v in kwargs.items() if k == "workdir"})
  raise ValueError(f"Unknown OTA_SANDBOX={kind!r} (expected gvisor|local).")
