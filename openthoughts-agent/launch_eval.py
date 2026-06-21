# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint: evaluate an SFT'd Qwen3-8B agent on OpenThoughts Terminal-Bench.

For each TB-dev task we build its environment image, run it under gVisor (runsc),
let the Terminus-2 agent loop drive the SFT'd policy to issue shell commands in
the sandbox, then run the task's grader (`tests/test.sh`) for pass/fail. Requires
the custom task image (`docker/Dockerfile.agent-task`, which ships runsc+docker)
selected via `iris job run --task-image ...`, and the SFT checkpoint in CKPT_DIR.

Config via env:
  * ``AGENT_MODEL`` (qwen3-8b), ``CKPT_DIR`` (required: the SFT checkpoint root).
  * ``TASK_LIMIT`` (unset = all TB-dev tasks; set small to start).
  * ``MAX_TURNS`` (20), ``COMMAND_TIMEOUT`` (60), ``MAX_NEW_TOKENS`` (1024).
  * ``TP`` (1), ``MAX_PROMPT_LEN`` (8192), ``TEMPERATURE`` (0.2).
  * ``OTA_SANDBOX`` (gvisor) -- set ``local`` only for harness debugging.

Submit (custom image + checkpoint):

    uv run iris --cluster=marin job run --no-wait \
      --task-image ghcr.io/<org>/openthoughts-agent-task:latest \
      --tpu v6e-8 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 16 --memory 200GB --disk 200GB --max-retries 1 --job-name ota-eval \
      -e HF_TOKEN "$HF_TOKEN" \
      -e CKPT_DIR gs://marin-us-central2/openthoughts-agent/qwen3-8b-agent-sft \
      -e TASK_LIMIT 10 -- python launch_eval.py
"""

import json
import os
import traceback

import jax
from huggingface_hub import snapshot_download

from eval.agent_loop import run_episode
from eval.grade import grade_task
from eval.model_serving import make_tunix_model_fn
from eval.sandbox import GvisorContainerSandbox, build_image
from eval.tb_tasks import load_tb_tasks
from models.checkpoint import restore_sft_model
from models.registry import get_model_spec
from training.common import build_mesh, init_distributed


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def main() -> None:
  init_distributed()  # must precede any jax call (orbax multi-host barriers)
  model_name = os.environ.get("AGENT_MODEL", "qwen3-8b")
  checkpoint_dir = os.environ["CKPT_DIR"]
  task_limit = os.environ.get("TASK_LIMIT")
  task_limit = int(task_limit) if task_limit else None
  max_turns = int(os.environ.get("MAX_TURNS", "20"))
  command_timeout = float(os.environ.get("COMMAND_TIMEOUT", "60"))
  max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
  max_prompt_len = int(os.environ.get("MAX_PROMPT_LEN", "8192"))
  temperature = float(os.environ.get("TEMPERATURE", "0.2"))
  tp = int(os.environ.get("TP", "1"))

  spec = get_model_spec(model_name)
  base_dir = _ensure_model(spec.repo, os.environ.get("AGENT_MODEL_DIR") or f"./{spec.name}")
  print(f"[ota-eval] jax {jax.__version__} devices={jax.device_count()} ckpt={checkpoint_dir}", flush=True)

  mesh = build_mesh(tp=tp)
  tokenizer = spec.load_tokenizer(base_dir)
  model = restore_sft_model(base_dir, checkpoint_dir, mesh=mesh)
  model_fn = make_tunix_model_fn(
      model, tokenizer, mesh,
      max_prompt_length=max_prompt_len,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
  )

  tasks = load_tb_tasks(limit=task_limit)
  print(f"[ota-eval] {len(tasks)} TB tasks to evaluate", flush=True)

  records = []
  for i, task in enumerate(tasks):
    print(f"[ota-eval] ({i+1}/{len(tasks)}) task={task.task_id}", flush=True)
    rec = {"task_id": task.task_id, "solved": False, "score": 0.0, "error": None}
    sandbox = None
    try:
      build = build_image(task.environment_dir, task.image_tag)
      if build.exit_code != 0:
        rec["error"] = f"image build failed: {build.stderr[-500:]}"
        records.append(rec)
        continue
      sandbox = GvisorContainerSandbox(task.image_tag)
      episode = run_episode(
          model_fn, sandbox, task.instruction,
          max_turns=max_turns, command_timeout=command_timeout,
      )
      grade = grade_task(sandbox, task)
      rec.update(
          solved=grade.solved,
          score=grade.score,
          turns=episode.turns,
          parse_failures=episode.parse_failures,
          detail=grade.detail,
      )
    except Exception as e:  # one task's failure must not kill the sweep
      rec["error"] = f"{type(e).__name__}: {e}"
      print(f"[ota-eval]   TRACE {task.task_id}:\n{traceback.format_exc()}", flush=True)
    finally:
      if sandbox is not None:
        sandbox.close()
    print(f"[ota-eval]   -> {json.dumps(rec)}", flush=True)
    records.append(rec)

  n = len(records)
  solved = sum(1 for r in records if r["solved"])
  mean_score = sum(r["score"] for r in records) / max(n, 1)
  print(f"[ota-eval] ===== RESULTS =====", flush=True)
  print(f"[ota-eval] solved {solved}/{n} = {solved/max(n,1):.3f} | mean_score={mean_score:.3f}", flush=True)
  print(f"[ota-eval] PER_TASK_JSON {json.dumps(records)}", flush=True)


if __name__ == "__main__":
  main()
