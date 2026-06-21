# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live gVisor (runsc) validation inside an iris TPU task.

TPU tasks run ``--privileged`` (iris adds it for accelerators), which is what
rootful gVisor + a task-local dockerd need. This smoke bootstraps the SAME
binaries the custom task image (`docker/Dockerfile.agent-task`) bakes in --
Docker's static binaries + runsc -- at runtime, then exercises the real
`eval/sandbox.py` paths end-to-end so we validate the mechanism without first
publishing the custom image to a registry:

  1. install docker + runsc, register the runsc Docker runtime,
  2. start dockerd (eval.sandbox.ensure_dockerd),
  3. run a container under runsc and PROVE gVisor isolation by comparing the
     kernel `uname -r` under runsc (gVisor's emulated kernel) vs runc (the host
     kernel) -- a different kernel means gVisor is interposing on syscalls,
  4. drive a GvisorContainerSandbox (exec + copy_in), the exact class the eval
     harness uses per Terminal-Bench task.

Submit on the smallest TPU slice (privileged, fast to schedule):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 60GB --disk 60GB --max-retries 1 --job-name ota-gvisor-smoke \
      -- python -m eval.gvisor_smoke
"""

import os
import shutil
import tarfile
import urllib.request

from eval.sandbox import (
    GvisorContainerSandbox,
    _run,
    ensure_dockerd,
)

DOCKER_VERSION = "27.3.1"
DOCKER_URL = f"https://download.docker.com/linux/static/stable/x86_64/docker-{DOCKER_VERSION}.tgz"
RUNSC_BASE = "https://storage.googleapis.com/gvisor/releases/release/latest/x86_64"
BIN_DIR = "/usr/local/bin"
DAEMON_JSON = (
    '{"runtimes":{"runsc":{"path":"/usr/local/bin/runsc",'
    '"runtimeArgs":["--platform=ptrace","--network=sandbox"]}}}'
)


def _log(msg: str) -> None:
  print(f"[gvisor-smoke] {msg}", flush=True)


def bootstrap() -> None:
  """Installs docker static binaries + runsc and registers the runsc runtime."""
  if shutil.which("docker") is None:
    _log(f"downloading docker {DOCKER_VERSION} static binaries")
    urllib.request.urlretrieve(DOCKER_URL, "/tmp/docker.tgz")
    with tarfile.open("/tmp/docker.tgz") as t:
      t.extractall("/tmp/docker-extract")
    for fn in os.listdir("/tmp/docker-extract/docker"):
      dst = os.path.join(BIN_DIR, fn)
      shutil.copy(os.path.join("/tmp/docker-extract/docker", fn), dst)
      os.chmod(dst, 0o755)
  if shutil.which("runsc") is None:
    _log("downloading runsc")
    urllib.request.urlretrieve(f"{RUNSC_BASE}/runsc", os.path.join(BIN_DIR, "runsc"))
    os.chmod(os.path.join(BIN_DIR, "runsc"), 0o755)
  os.makedirs("/etc/docker", exist_ok=True)
  with open("/etc/docker/daemon.json", "w") as f:
    f.write(DAEMON_JSON)


def main() -> None:
  _log(f"uid={os.getuid()} (privileged TPU task expected to be root)")
  bootstrap()

  v = _run(["runsc", "--version"], timeout=30)
  _log(f"runsc --version -> exit={v.exit_code}\n{v.stdout}{v.stderr}")

  ensure_dockerd(timeout=90)
  _log("dockerd is up")

  # gVisor isolation proof: kernel under runsc vs runc.
  host = _run(["uname", "-r"], timeout=15).stdout.strip()
  _log(f"host kernel (task VM): {host}")
  runc = _run(["docker", "run", "--rm", "alpine", "uname", "-r"], timeout=300)
  _log(f"runc container kernel: {runc.stdout.strip()!r} (exit={runc.exit_code}) {runc.stderr[-200:]}")
  gv = _run(["docker", "run", "--rm", "--runtime=runsc", "alpine", "uname", "-r"], timeout=300)
  _log(f"runsc container kernel: {gv.stdout.strip()!r} (exit={gv.exit_code}) {gv.stderr[-300:]}")

  procver = _run(
      ["docker", "run", "--rm", "--runtime=runsc", "alpine", "sh", "-c", "cat /proc/version; dmesg 2>/dev/null | head -3"],
      timeout=120,
  )
  _log(f"runsc /proc/version + dmesg:\n{procver.stdout}")

  isolated = gv.exit_code == 0 and gv.stdout.strip() and gv.stdout.strip() != runc.stdout.strip()
  _log(f"GVISOR ISOLATION: {'CONFIRMED (runsc kernel != host kernel)' if isolated else 'NOT CONFIRMED'}")

  # Exercise the production sandbox class on a stock image.
  _log("exercising GvisorContainerSandbox(alpine)")
  sb = GvisorContainerSandbox("alpine", workdir="/root")
  try:
    r = sb.exec("echo from-sandbox && uname -r && id")
    _log(f"sandbox.exec -> exit={r.exit_code}\n{r.stdout}{r.stderr[-200:]}")
    # copy_in a local file then read it back inside the sandbox.
    with open("/tmp/ota_probe.txt", "w") as f:
      f.write("hello-from-host\n")
    cp = sb.copy_in("/tmp/ota_probe.txt", "/root/probe.txt")
    rb = sb.exec("cat /root/probe.txt")
    _log(f"sandbox.copy_in exit={cp.exit_code}; read-back={rb.stdout.strip()!r}")
  finally:
    sb.close()

  _log("DONE")


if __name__ == "__main__":
  main()
