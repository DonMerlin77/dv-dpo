"""
mimir/auto_red_team.py

Autonomous pair generator. No external API calls — Mimir generates, Mimir judges.

Flow per question:
  1. Run Mimir → get response
  2. Heuristic detector → if fails, identify failure type
  3. Re-run Mimir with constraint hint → get corrected response
  4. Validate correction with detector → skip if still fails
  5. Write {prompt, chosen, rejected, meta} pair

Usage:
  python3 mimir/auto_red_team.py --questions mimir/dvdpo_run_questions.jsonl
  python3 mimir/auto_red_team.py --questions mimir/dvdpo_run_questions.jsonl --out mimir/auto_pairs.jsonl --target 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mimir.failure_detector import detect, heuristic_check

MIMIR_URL = "https://tomsem77--mimir-orlog-serve-serve.modal.run/generate"

# Constraint hints injected into the prompt per failure type.
# These guide Mimir to fix the specific flaw without changing anything else.
CONSTRAINT_HINTS = {
    "coherence": (
        "IMPORTANT: Your VERDICT must take a clear position. "
        "Every ACTION step must follow directly from that position — "
        "they cannot contradict the VERDICT."
    ),
    "grounding": (
        "IMPORTANT: Your BLIND SPOT must only identify something "
        "derivable from what was actually written in the question. "
        "Do not introduce facts, people, or circumstances not mentioned."
    ),
    "context_sobriety": (
        "IMPORTANT: This person is sober / in recovery. "
        "Do not mention alcohol, drinks, or drinking limits in your response."
    ),
    "context_relationship_status": (
        "IMPORTANT: Note the person's stated relationship status. "
        "Do not suggest they pursue new relationships."
    ),
    "context_no_money": (
        "IMPORTANT: This person has stated they have no money or are in debt. "
        "Do not recommend investing until debt is addressed."
    ),
    "format": (
        "IMPORTANT: Your response must include all four sections in order: "
        "VERDICT, KEY TENSION, BLIND SPOT, ACTION."
    ),
}

ORLOG_FORMAT = """VERDICT: <one sentence, takes a clear position>

KEY TENSION: <the real conflict, 8 words or fewer>

BLIND SPOT: <one sentence, non-obvious, derivable from what was actually said>

ACTION:
- <verb-first, under 15 words, must follow from VERDICT>
- <second step>
- <third step>"""


def query(url: str, question: str, hint: str = "", timeout: int = 90) -> str:
    prompt = f"{hint}\n\n{question}".strip() if hint else question
    try:
        r = requests.post(url, json={"question": prompt}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def process_question(question: str, url: str) -> dict | None:
    # Step 1: original response
    rejected = query(url, question)
    if rejected.startswith("ERROR") or not rejected:
        return None

    # Step 2: detect failures (heuristic only — free)
    result = heuristic_check(question, rejected)
    if result["pass"]:
        return None  # no failure to fix

    failures = result["failures"]
    failure_labels = [f["label"] for f in failures]

    # Step 3: build constraint hint from first failure
    hint = CONSTRAINT_HINTS.get(failure_labels[0], "")
    if not hint:
        return None

    # Step 4: generate corrected response with hint
    chosen = query(url, question, hint=hint)
    if chosen.startswith("ERROR") or not chosen or chosen == rejected:
        return None

    # Step 5: validate correction — skip if chosen still fails
    validation = heuristic_check(question, chosen)
    if not validation["pass"]:
        return None

    return {
        "prompt": question,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "failures": failure_labels,
            "source": "auto_red_team",
        },
    }


def load_questions(path: Path) -> list[str]:
    questions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            q = obj.get("question") or obj.get("prompt") or ""
            if q:
                questions.append(q)
        except json.JSONDecodeError:
            questions.append(line)
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="JSONL of questions")
    parser.add_argument("--out", default="mimir/auto_pairs.jsonl")
    parser.add_argument("--url", default=MIMIR_URL)
    parser.add_argument("--target", type=int, default=0, help="Stop after N pairs (0 = run all)")
    parser.add_argument("--dry", action="store_true", help="Print failures without writing pairs")
    args = parser.parse_args()

    q_path = ROOT / args.questions
    questions = load_questions(q_path)
    out_path = ROOT / args.out

    print(f"Questions: {len(questions)}")
    print(f"Output:    {out_path}")
    print(f"Target:    {args.target or 'all'}")
    print()

    written = 0
    tested = 0
    failed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as f:
        for i, q in enumerate(questions):
            if args.target and written >= args.target:
                break

            tested += 1
            print(f"[{i+1}/{len(questions)}] {q[:70]}")

            pair = process_question(q, args.url)

            if pair is None:
                print(f"  — pass or unfixable")
            else:
                failed += 1
                print(f"  FAIL {pair['meta']['failures']} → pair generated")
                if not args.dry:
                    f.write(json.dumps(pair) + "\n")
                    written += 1
                else:
                    print(f"  chosen[:80]: {pair['chosen'][:80]}")

            time.sleep(0.3)

    print(f"\n{'='*50}")
    print(f"Tested:  {tested}")
    print(f"Flagged: {failed}")
    print(f"Pairs:   {written} → {out_path}")
    if tested:
        print(f"Failure rate: {failed/tested:.0%}")


if __name__ == "__main__":
    main()
