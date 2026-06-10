"""
mimir/auto_train_loop.py

Full autonomous cycle: detect failures → generate pairs → train → deploy.

Steps:
  1. Run auto_red_team on question pool → accumulate pairs into auto_pairs.jsonl
  2. Merge auto_pairs.jsonl with base training set → new versioned JSONL
  3. Kick off Modal training job
  4. Quantize + merge the new checkpoint
  5. Update mimir_orlog_serve.py MERGED path → redeploy
  6. Run eval on held-out questions → report pass rate

Usage:
  python3 mimir/auto_train_loop.py --questions mimir/dvdpo_run_questions.jsonl
  python3 mimir/auto_train_loop.py --questions mimir/dvdpo_run_questions.jsonl --min-pairs 30 --version orlog_v5
  python3 mimir/auto_train_loop.py --skip-redteam --version orlog_v5  # use existing auto_pairs.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

SERVE_FILE = ROOT / "orlog/app/mimir_orlog_serve.py"
EVAL_SCRIPT = ROOT / "mimir/eval_orlog_v4.py"
BASE_PAIRS   = ROOT / "mimir/orlog_v4_pairs.jsonl"
AUTO_PAIRS   = ROOT / "mimir/auto_pairs.jsonl"
MIMIR_URL    = "https://tomsem77--mimir-orlog-serve-serve.modal.run/generate"


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def build_training_set(version: str) -> Path:
    out = ROOT / f"mimir/{version}_pairs.jsonl"
    base_count = count_lines(BASE_PAIRS)
    auto_count  = count_lines(AUTO_PAIRS)
    total = base_count + auto_count

    print(f"\nBuilding training set: {base_count} base + {auto_count} auto = {total} pairs")

    lines = []
    if BASE_PAIRS.exists():
        lines += [l for l in BASE_PAIRS.read_text().splitlines() if l.strip()]
    if AUTO_PAIRS.exists():
        lines += [l for l in AUTO_PAIRS.read_text().splitlines() if l.strip()]

    out.write_text("\n".join(lines) + "\n")
    print(f"Training set → {out}")
    return out


def update_serve_version(new_version: str):
    text = SERVE_FILE.read_text()
    import re
    old = re.search(r'MERGED = "/checkpoints/(\S+)"', text)
    if old:
        old_ver = old.group(1)
        text = text.replace(
            f'MERGED = "/checkpoints/{old_ver}"',
            f'MERGED = "/checkpoints/{new_version}_merged"',
        )
        text = text.replace(
            f'"mimir-orlog-{old_ver.replace("_merged","")}"',
            f'"mimir-orlog-{new_version}"',
        )
        SERVE_FILE.write_text(text)
        print(f"Updated serve file: {old_ver} → {new_version}_merged")
    else:
        print("WARNING: could not find MERGED path in serve file — update manually")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="mimir/dvdpo_run_questions.jsonl")
    parser.add_argument("--version", default="orlog_v5")
    parser.add_argument("--min-pairs", type=int, default=20,
                        help="Minimum new auto pairs before training (skip if below)")
    parser.add_argument("--skip-redteam", action="store_true",
                        help="Skip auto_red_team, use existing auto_pairs.jsonl")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    args = parser.parse_args()

    venv = "source venv/bin/activate &&"

    # Step 1: red team
    if not args.skip_redteam:
        print("\n=== Step 1: Auto Red Team ===")
        run(f"cd {ROOT} && {venv} python3 mimir/auto_red_team.py "
            f"--questions {args.questions} --out mimir/auto_pairs.jsonl", check=False)

    auto_count = count_lines(AUTO_PAIRS)
    print(f"\nauto_pairs.jsonl: {auto_count} pairs")

    if auto_count < args.min_pairs:
        print(f"Only {auto_count} auto pairs — below threshold of {args.min_pairs}. Stopping.")
        print("Run more questions or lower --min-pairs to proceed.")
        sys.exit(0)

    # Step 2: build training set
    print("\n=== Step 2: Build Training Set ===")
    training_set = build_training_set(args.version)

    # Step 3: train
    if not args.skip_train:
        print("\n=== Step 3: Train ===")
        run(f"cd {ROOT} && {venv} modal run mimir/modal_train.py::train_orlog_dpo "
            f"--pairs-file mimir/{args.version}_pairs.jsonl --model {args.version}")

        # Step 4: quantize
        print("\n=== Step 4: Quantize ===")
        run(f"cd {ROOT} && {venv} modal run mimir/modal_train.py::run_quantize --model {args.version}")

    # Step 5: deploy
    if not args.skip_deploy:
        print("\n=== Step 5: Deploy ===")
        update_serve_version(args.version)
        run(f"cd {ROOT} && {venv} modal deploy orlog/app/mimir_orlog_serve.py")

    # Step 6: eval
    print("\n=== Step 6: Eval ===")
    run(f"cd {ROOT} && {venv} python3 mimir/eval_orlog_v4.py "
        f"--v4-url {MIMIR_URL} --out mimir/eval_results/{args.version}_eval.json", check=False)

    print(f"\n{'='*60}")
    print(f"Cycle complete. Version: {args.version}")
    print(f"Eval results: mimir/eval_results/{args.version}_eval.json")


if __name__ == "__main__":
    main()
