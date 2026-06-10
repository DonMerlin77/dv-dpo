"""
mimir/unanimous_sampler.py

Audits high-confidence unanimous council responses for shared bias.

The silent failure class: all 3 council voices agree confidently but are wrong together.
These never produce a revision signal, so they slip through DV-DPO undetected.

Fix (per Commercial_Eagle_693): sample the unanimous set weighted toward HIGHEST confidence.
The proudest responses are most likely to carry shared prior from the same training data.

Usage:
  python3 mimir/unanimous_sampler.py
  python3 mimir/unanimous_sampler.py --n 20 --out mimir/unanimous_audit.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
HAIKU = "claude-haiku-4-5-20251001"

DPO_PAIRS    = ROOT / "mimir" / "deliberation_dpo_pairs.jsonl"
HISTORY      = ROOT / "orlog" / "local_history.jsonl"
DEFAULT_OUT  = ROOT / "mimir" / "unanimous_audit.jsonl"


JUDGE_SYSTEM = """You are a strict quality judge for AI decision-reasoning responses.
You are checking for shared bias — cases where the model confidently gave an answer that sounds reasonable but is subtly wrong, incomplete, or misleading.

Focus on:
1. GROUNDING — Does the response introduce facts or assumptions NOT in the question?
2. COHERENCE — Does the VERDICT align with the ACTION steps?
3. BLIND SPOT QUALITY — Is the blind spot genuinely non-obvious, or is it generic advice anyone would give?
4. CONFIDENCE CALIBRATION — Does the model express certainty where genuine uncertainty exists?

Reply in this exact format:
VERDICT: PASS or FAIL
ISSUE: one sentence describing the problem (or "none" if PASS)
CONFIDENCE: how confident are you in this judgment? HIGH / MEDIUM / LOW"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def score_confidence(pair: dict) -> float:
    distance = pair.get("distance", 0.5)
    was_revised = pair.get("was_revised", False)
    if was_revised:
        return 0.0
    return 1.0 - distance


def load_candidates() -> list[dict]:
    candidates = []

    pairs = load_jsonl(DPO_PAIRS)
    for p in pairs:
        if not p.get("was_revised", False):
            candidates.append({
                "question": p.get("prompt", ""),
                "response": p.get("chosen", ""),
                "confidence": score_confidence(p),
                "source": "deliberation_pairs",
            })

    history = load_jsonl(HISTORY)
    for h in history:
        if h.get("failures"):
            continue
        candidates.append({
            "question": h.get("question", ""),
            "response": h.get("response", ""),
            "confidence": 0.7,
            "source": "local_history",
        })

    return candidates


def weighted_sample(candidates: list[dict], n: int) -> list[dict]:
    if not candidates:
        return []
    weights = [max(c["confidence"], 0.01) for c in candidates]
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen = random.choices(candidates, weights=probs, k=min(n, len(candidates)))
    chosen.sort(key=lambda x: x["confidence"], reverse=True)
    return chosen


def judge(question: str, response: str) -> dict:
    prompt = f"""QUESTION: {question}

RESPONSE:
{response}"""

    try:
        msg = client.messages.create(
            model=HAIKU,
            max_tokens=150,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        result = {"raw": raw, "verdict": "UNKNOWN", "issue": "", "judge_confidence": "LOW"}
        for line in raw.splitlines():
            if line.startswith("VERDICT:"):
                result["verdict"] = "FAIL" if "FAIL" in line else "PASS"
            elif line.startswith("ISSUE:"):
                result["issue"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                result["judge_confidence"] = line.split(":", 1)[1].strip()
        return result
    except Exception as e:
        return {"verdict": "ERROR", "issue": str(e), "judge_confidence": "LOW", "raw": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of responses to audit (default: 10)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output file for failures")
    args = parser.parse_args()

    candidates = load_candidates()
    if not candidates:
        print("No candidates found. Run deliberation_dpo.py or local_app.py first to generate history.")
        return

    sample = weighted_sample(candidates, args.n)
    print(f"Candidates: {len(candidates)} | Sampling: {len(sample)} (weighted toward highest confidence)\n")

    failures = []
    for i, c in enumerate(sample, 1):
        q_short = c["question"][:60]
        print(f"[{i}/{len(sample)}] conf={c['confidence']:.2f} [{c['source']}] {q_short}...")
        result = judge(c["question"], c["response"])
        status = "FAIL" if result["verdict"] == "FAIL" else "ok"
        print(f"  → {status} ({result['judge_confidence']}) {result['issue'][:80] if result['issue'] != 'none' else ''}")

        if result["verdict"] == "FAIL":
            failures.append({
                "question": c["question"],
                "response": c["response"],
                "confidence": c["confidence"],
                "source": c["source"],
                "issue": result["issue"],
                "judge_confidence": result["judge_confidence"],
                "raw": result["raw"],
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for fail in failures:
            f.write(json.dumps(fail) + "\n")

    print(f"\n{'='*50}")
    print(f"Audited:  {len(sample)}")
    print(f"Failed:   {len(failures)}")
    print(f"Rate:     {len(failures)/len(sample):.1%}")
    if failures:
        print(f"Written:  {out_path}")
        print("\nAdd failed pairs to auto_pairs.jsonl to feed the training loop.")


if __name__ == "__main__":
    main()
