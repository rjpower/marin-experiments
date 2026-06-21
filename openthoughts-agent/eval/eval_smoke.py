# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-free end-to-end eval integration smoke (run on a privileged TPU task).

Proves the non-policy half of the eval -- `build_image` -> `GvisorContainerSandbox`
-> oracle solve -> `grade_task` -- works against REAL OpenThoughts-TB-dev tasks
under gVisor, without needing the 8B checkpoint. Each TB task ships an oracle
`solution/solve.sh`; we build the task image, run the oracle inside the gVisor
sandbox, then grade. Expected result: score 1.0 (the oracle solves the task), so
any task that does NOT score 1.0 points at a harness bug (build, sandbox, copy,
WORKDIR, or grader path), not a model failure.

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 80GB --disk 100GB --max-retries 1 --job-name ota-eval-smoke \
      -e HF_TOKEN "$HF_TOKEN" -e TASK_LIMIT 3 -- python -m eval.eval_smoke
"""

import json
import os
import time

from eval.grade import grade_task
from eval.sandbox import GvisorContainerSandbox, build_image, ensure_sandbox_runtime
from eval.tb_tasks import load_tb_tasks


def _log(msg: str) -> None:
  print(f"[eval-smoke] {msg}", flush=True)


def main() -> None:
  # No-op on the custom task image; installs docker+runsc on the stock iris image.
  ensure_sandbox_runtime()
  limit = int(os.environ.get("TASK_LIMIT", "3"))
  tasks = load_tb_tasks(limit=limit)
  _log(f"{len(tasks)} tasks to oracle-grade")

  records = []
  for i, task in enumerate(tasks):
    rec = {"task_id": task.task_id, "built": False, "oracle_exit": None,
           "solved": False, "score": 0.0, "detail": None}
    _log(f"({i+1}/{len(tasks)}) task={task.task_id}")
    sandbox = None
    try:
      t0 = time.monotonic()
      build = build_image(task.environment_dir, task.image_tag)
      rec["build_secs"] = round(time.monotonic() - t0, 1)
      if build.exit_code != 0:
        rec["detail"] = f"build failed: {build.stderr[-400:]}"
        records.append(rec)
        _log(f"  BUILD FAILED ({rec['build_secs']}s): {build.stderr[-300:]}")
        continue
      rec["built"] = True
      _log(f"  built in {rec['build_secs']}s; starting gVisor sandbox")
      sandbox = GvisorContainerSandbox(task.image_tag)

      solve = os.path.join(task.root, "solution", "solve.sh")
      if os.path.isfile(solve):
        sandbox.copy_in(solve, "/tmp/solve.sh")
        r = sandbox.exec("bash /tmp/solve.sh", timeout=task.agent_timeout_sec)
        rec["oracle_exit"] = r.exit_code
        _log(f"  oracle solve.sh exit={r.exit_code}; stdout tail={r.stdout[-200:]!r}")
      else:
        _log("  no oracle solve.sh (skipping solve, grading bare image)")

      g = grade_task(sandbox, task)
      rec.update(solved=g.solved, score=g.score, detail=g.detail,
                 test_exit=g.test_exit_code)
      _log(f"  GRADE solved={g.solved} score={g.score} ({g.detail})")
    except Exception as e:  # one task must not kill the smoke
      rec["detail"] = f"{type(e).__name__}: {e}"
      _log(f"  ERROR {rec['detail']}")
    finally:
      if sandbox is not None:
        sandbox.close()
    records.append(rec)

  solved = sum(1 for r in records if r["solved"])
  built = sum(1 for r in records if r["built"])
  _log("===== EVAL-SMOKE RESULTS =====")
  _log(f"built {built}/{len(records)} | oracle-solved {solved}/{len(records)} "
       f"(want solved==built; <build means a harness bug, not a model issue)")
  _log(f"PER_TASK_JSON {json.dumps(records)}")


if __name__ == "__main__":
  main()
