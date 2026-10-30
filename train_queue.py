"""Finish the interrupted PPO runs one after another, unattended.

Each launch continues from the newest checkpoint of a run into a fresh ``runs/<name>_resumeN``
directory: resuming in place would overwrite the monitor logs and the eval history and, because
EvalCallback forgets its best score, replace ``best/best_model.zip`` on its first evaluation.
Safe to start again after an interruption (or a reboot); finished runs are skipped.

    .venv\\Scripts\\python.exe train_queue.py
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_STEPS = 5_000_000
# Most promising first. Worker counts as in the original runs; 8 GB of RAM rules out running both at once.
QUEUE = (
    ("ppo_residual", "residual", 4),
    ("ppo_crosswind", "absolute", 8),
)
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def latest_checkpoint(name: str) -> tuple[int, Path]:
    found = []
    for run in (ROOT / "runs").glob(f"{name}*"):
        for ckpt in (run / "checkpoints").glob("ppo_landing_*_steps.zip"):
            steps = int(re.search(r"_(\d+)_steps", ckpt.name).group(1))
            found.append((steps, ckpt))
    return max(found, key=lambda item: item[0])


def next_run_dir(name: str) -> Path:
    numbers = [int(m.group(1)) for p in (ROOT / "runs").glob(f"{name}_resume*") if (m := re.search(r"_resume(\d+)$", p.name))]
    return ROOT / "runs" / f"{name}_resume{max(numbers, default=0) + 1}"


def finished(name: str) -> bool:
    return any((ROOT / "runs").glob(f"{name}_resume*/final_model.zip"))


def main() -> int:
    # Keep the machine awake and leave the foreground (the 3D console) priority, for this process tree only.
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # pseudo-handle -1: truncated to int it is rejected
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
        print("[queue] could not lower the process priority", flush=True)
    status = 0
    for name, action_mode, n_envs in QUEUE:
        if finished(name):
            print(f"[queue] {name}: already finished", flush=True)
            continue
        steps, ckpt = latest_checkpoint(name)
        run_dir = next_run_dir(name)
        # learn() adds the checkpoint's step count to this number, so pass only what remains.
        remaining = TARGET_STEPS - steps
        print(f"[queue] {name}: {steps} -> {TARGET_STEPS} steps from {ckpt.name}, logging to {run_dir.name}", flush=True)
        with (ROOT / f"{run_dir.name}.log").open("w", encoding="utf-8") as log:
            code = subprocess.call(
                [sys.executable, "-m", "src.agents.train_ppo", "--run-dir", str(run_dir.relative_to(ROOT)),
                 "--resume-from", str(ckpt.relative_to(ROOT)), "--total-timesteps", str(remaining),
                 "--action-mode", action_mode, "--n-envs", str(n_envs)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        print(f"[queue] {name}: exit code {code}", flush=True)
        status = status or code
    return status


if __name__ == "__main__":
    sys.exit(main())
