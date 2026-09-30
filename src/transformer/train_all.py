"""Run PhoBERT training for both tasks sequentially.

Logs to results/transformer/train_log.txt — check that file for progress.
Uses -u flag for unbuffered output so progress is visible in real time.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = BASE_DIR / "results" / "transformer" / "train_log.txt"


def run_task(task: str):
    cmd = [
        sys.executable, "-u",  # unbuffered
        str(BASE_DIR / "src" / "transformer" / "train_task.py"),
        "--task", task,
        "--max-length", "32",
    ]
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"START {task} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"CMD: {' '.join(cmd)}\n")
        log.write(f"{'='*60}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\nEXIT {task} — code={proc.returncode} "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.flush()


def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("Usage: python src/transformer/train_all.py")
        print("Runs train_task.py --task sentiment and --task topic sequentially,")
        print(f"logging to {LOG_FILE.relative_to(BASE_DIR)}.")
        print("For per-task options, run: python src/transformer/train_task.py --help")
        return
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    for task in ("sentiment", "topic"):
        run_task(task)


if __name__ == "__main__":
    main()
