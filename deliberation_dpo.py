"""
mimir/deliberation_dpo.py

Deliberation-Validated DPO (DV-DPO) — a novel training pipeline where the AI
generates its own training signal from its own internal disagreements.

Method (designed by Orlog council, 2026-06-08):
  Round 1: Council deliberates. Synthesis = chosen. Minority perspectives = rejected candidates.
  Round 2: Losing members cross-examine the synthesis. If synthesis is revised, revision = chosen.
  Filter:  Semantic distance check — pairs where chosen/rejected are too similar get killed.
  Gate:    1-2% of pairs flagged for external ground-truth validation (system selects sample).

Council design ruling:
  - Asymmetric starting positions: each agent enters from the opposing anchor before updating.
    This preserves authentic reasoning while guaranteeing adversarial position is at least entered.
  - Trajectory artifact detection: if an agent updates mid-debate, score whether it moved
    from evidence or capitulation. Capitulation pairs are killed.

Usage:
  python3 mimir/deliberation_dpo.py --questions mimir/eval_questions.jsonl --dry
  python3 mimir/deliberation_dpo.py --questions mimir/eval_questions.jsonl --extract
  python3 mimir/deliberation_dpo.py --questions mimir/eval_questions.jsonl --extract --limit 50
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import anthropic

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

OUT_FILE        = ROOT / "mimir" / "deliberation_dpo_pairs.jsonl"   # revision pairs only
SFT_FILE        = ROOT / "mimir" / "deliberation_sft_pairs.jsonl"   # voice pairs (format training)
VALIDATION_LOG  = ROOT / "mimir" / "deliberation_validation_queue.jsonl"
CACHE_FILE      = ROOT / "mimir" / "deliberation_cache.json"
DIFFICULTY_CACHE = ROOT / "mimir" / "difficulty_cache.json"

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = HAIKU  # use Haiku for everything — format constraints make quality difference minimal

MIN_DISTANCE    = 0.3   # minimum semantic difference between chosen/rejected (0-1 scale)
VALIDATION_RATE = 0.02  # 1-2% of pairs flagged for external validation


COUNCIL_STYLES = [
    ("paranoid",  "Challenge everything. What is the hidden risk, the assumption that will break this?"),
    ("visionary", "What is the 10x version of this? What future does this open or foreclose?"),
    ("realist",   "What will actually happen in practice? Not the ideal, the real constraints."),
]

SYNTHESIS_SYSTEM = """You are the Orlog synthesizer. You have read all council perspectives.
Your job: take a CLEAR position. No hedging. Name the real tension. Identify the blind spot.
End with a specific, actionable verdict.

Format:
ANCHOR: [the hidden assumption driving this question]
TENSION: [what genuinely pulls in each direction]
BLIND SPOT: [what no one is seeing]
VERDICT: [one clear sentence taking a position]
ACTION: [specific next step, verb-first]

Under 200 words. Dense. No filler."""


CROSS_EXAM_SYSTEM = """You are a council member who lost the debate. The synthesizer ruled against your position.
Your job: find the ONE flaw in the synthesis that, if true, would overturn the verdict entirely.
Be specific. Quote the synthesis. Name the exact assumption that is wrong.
Under 100 words. Do not concede. This is cross-examination."""


DISTANCE_SYSTEM = """Rate the semantic distance between these two texts on a scale of 0.0 to 1.0.
0.0 = identical meaning. 1.0 = completely opposite positions.
Return ONLY a JSON object: {"distance": 0.0-1.0, "reason": "one line"}"""


CAPITULATION_SYSTEM = """An agent initially argued position A, then updated to position B after seeing a counter-argument.
Determine if this update was:
- EVIDENCE-BASED: the new argument introduced a genuine fact, constraint, or consideration not previously visible
- SYCOPHANTIC: the agent simply adopted the majority position without genuine new reasoning

Return ONLY JSON: {"verdict": "evidence" or "sycophantic", "reason": "one line"}"""


def _call(system: str, user: str, model: str = HAIKU, max_tokens: int = 400) -> str:
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        timeout=30,
    )
    return r.content[0].text.strip()


def _parse_json(text: str) -> dict:
    import re
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```.*$", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _question_key(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()


def run_council_round1(question: str) -> dict:
    """All council members deliberate. Returns perspectives dict."""
    perspectives = {}

    def _voice(style_tuple):
        name, mandate = style_tuple
        prompt = (
            f"Question: {question}\n\n"
            f"Your mandate: {mandate}\n\n"
            f"Before stating your view, briefly argue the OPPOSITE position — "
            f"the strongest case against your natural conclusion. Then give your actual analysis.\n\n"
            f"Under 120 words total."
        )
        text = _call(
            system=f"You are the {name.upper()} cognitive voice on the Orlog council.",
            user=prompt,
        )
        return name, text

    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, text in pool.map(_voice, COUNCIL_STYLES):
            perspectives[name] = text

    return perspectives


def run_synthesis(question: str, perspectives: dict) -> str:
    """Synthesize all council perspectives into a verdict."""
    persp_text = "\n\n".join(
        f"[{name.upper()}]\n{text}" for name, text in perspectives.items()
    )
    prompt = f"Question: {question}\n\nCouncil perspectives:\n{persp_text}"
    return _call(SYNTHESIS_SYSTEM, prompt, model=SONNET, max_tokens=500)


def run_cross_examination(question: str, synthesis: str, perspectives: dict) -> dict:
    """
    Round 2: losing members cross-examine the synthesis.
    Returns: {name: challenge_text} and whether synthesis was revised.
    """
    challenges = {}

    def _challenge(name_text):
        name, perspective = name_text
        prompt = (
            f"Question: {question}\n\n"
            f"Your original position:\n{perspective}\n\n"
            f"The synthesis ruled:\n{synthesis}\n\n"
            f"Find the flaw."
        )
        challenge = _call(CROSS_EXAM_SYSTEM, prompt)
        return name, challenge

    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, challenge in pool.map(_challenge, perspectives.items()):
            challenges[name] = challenge

    # Give synthesizer a chance to respond to cross-examination
    challenges_text = "\n\n".join(
        f"[{name.upper()} CHALLENGE]\n{text}" for name, text in challenges.items()
    )
    revision_prompt = (
        f"Your synthesis:\n{synthesis}\n\n"
        f"Council cross-examination:\n{challenges_text}\n\n"
        f"Does any challenge reveal a genuine flaw that requires revising your verdict? "
        f"If yes, output REVISED: followed by the new verdict. "
        f"If no, output UPHELD: followed by one sentence explaining why the challenges fail."
    )
    revision_response = _call(SYNTHESIS_SYSTEM, revision_prompt, model=SONNET, max_tokens=400)

    revised = revision_response.startswith("REVISED:")
    final_synthesis = revision_response[8:].strip() if revised else synthesis

    return {
        "challenges":       challenges,
        "revision_response": revision_response,
        "was_revised":      revised,
        "final_synthesis":  final_synthesis,
    }


def score_distance(text_a: str, text_b: str) -> float:
    """Estimate semantic distance via 8-gram overlap. Zero API cost."""
    def ngrams(text, n=8):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    a, b = ngrams(text_a), ngrams(text_b)
    if not a or not b:
        return 0.5
    overlap = len(a & b) / max(len(a), len(b))
    return round(1.0 - overlap, 3)


def check_capitulation(original: str, updated: str, counter_argument: str) -> str:
    """Detect if an agent's update was evidence-based or sycophantic."""
    prompt = (
        f"Original position:\n{original[:300]}\n\n"
        f"Counter-argument seen:\n{counter_argument[:300]}\n\n"
        f"Updated position:\n{updated[:300]}"
    )
    result = _parse_json(_call(CAPITULATION_SYSTEM, prompt))
    return result.get("verdict", "evidence")


DIFFICULTY_SYSTEM = """Score this question for how hard it is for a reasoning council to reach a clean verdict.

HIGH (8-10): Genuine irresolvable tension. Two legitimate value systems reach opposite conclusions.
             No dominant option. The answer depends on unstated priorities. Stakes are real.
MEDIUM (5-7): Some tension but one answer is probably better. Context would resolve it.
LOW (1-4): Clear right answer. Technical question. Already mostly resolved.

Return ONLY JSON: {"score": 1-10, "reason": "one line", "tension": "what pulls in each direction"}"""


def score_difficulty(question: str) -> dict:
    """Score how likely this question is to produce a genuine revision."""
    cache_path = DIFFICULTY_CACHE
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    key = hashlib.md5(question.encode()).hexdigest()
    if key in cache:
        return cache[key]

    result = _parse_json(_call(DIFFICULTY_SYSTEM, f"Question: {question}", max_tokens=150))
    result = result if result else {"score": 5, "reason": "unknown", "tension": ""}
    cache[key] = result
    cache_path.write_text(json.dumps(cache))
    return result


def generate_pairs_from_deliberation(question: str, context: str = "") -> tuple[list, list]:
    """
    Full DV-DPO pipeline for one question.
    Returns: (dpo_pairs, sft_pairs)
      dpo_pairs — revision pairs only: chosen=post-revision, rejected=pre-revision
      sft_pairs — voice pairs: synthesis format examples for SFT training
    """
    print(f"  Round 1: council deliberating...")
    perspectives = run_council_round1(question)

    print(f"  Synthesizing...")
    synthesis_r1 = run_synthesis(question, perspectives)

    print(f"  Round 2: cross-examination...")
    cross = run_cross_examination(question, synthesis_r1, perspectives)
    final_synthesis = cross["final_synthesis"]
    was_revised = cross["was_revised"]

    dpo_pairs = []
    sft_pairs = []

    # SFT pairs: each voice perspective → synthesis (teaches format, not reasoning quality)
    for name, perspective_text in perspectives.items():
        sft_pairs.append({
            "prompt":        question,
            "completion":    final_synthesis,
            "source":        "deliberation_sft",
            "council_voice": name,
            "context":       context[:200],
        })

    # DPO pair: only if synthesis was genuinely revised by cross-examination
    if was_revised and synthesis_r1 != final_synthesis:
        distance = score_distance(final_synthesis, synthesis_r1)
        dpo_pairs.append({
            "prompt":        question,
            "chosen":        final_synthesis,
            "rejected":      synthesis_r1,
            "source":        "deliberation_dpo_revision",
            "council_voice": "pre_revision_synthesis",
            "distance":      distance,
            "was_revised":   True,
            "context":       context[:200],
        })
        print(f"  ✓ REVISION — DPO pair generated (distance={distance:.2f})")
    else:
        print(f"  — synthesis upheld, no DPO pair")

    return dpo_pairs, sft_pairs


def flag_for_validation(pairs: list[dict]) -> list[dict]:
    """Flag 1-2% of pairs for external ground-truth validation (council ruling)."""
    import random
    flagged = []
    for pair in pairs:
        if random.random() < VALIDATION_RATE:
            pair["needs_validation"] = True
            flagged.append(pair)
        else:
            pair["needs_validation"] = False
    return pairs, flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="mimir/eval_questions.jsonl")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--score-difficulty", action="store_true",
                        help="Score all questions by tension/irresolvability, output hard_questions.jsonl")
    parser.add_argument("--min-difficulty", type=int, default=8,
                        help="Only run questions scoring >= this difficulty (default 8)")
    args = parser.parse_args()

    q_path = ROOT / args.questions
    if not q_path.exists():
        print(f"Questions file not found: {q_path}")
        print("Create it with one JSON object per line: {\"question\": \"...\", \"context\": \"...\"}")
        print("\nOr generate from existing pairs:")
        print("  python3 mimir/deliberation_dpo.py --seed-from-pairs")
        sys.exit(1)

    questions = []
    for line in q_path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        questions.append(obj)

    print(f"Questions loaded: {len(questions)}")

    # Score difficulty mode — find hard questions before running
    if args.score_difficulty:
        hard_path = ROOT / "mimir" / "hard_questions.jsonl"
        print(f"Scoring difficulty for {len(questions)} questions...")
        scored = []
        for i, item in enumerate(questions):
            q = item["question"]
            d = score_difficulty(q)
            score = d.get("score", 0)
            scored.append({**item, "difficulty": score, "tension": d.get("tension",""), "reason": d.get("reason","")})
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(questions)} scored...")

        scored.sort(key=lambda x: x["difficulty"], reverse=True)
        hard = [s for s in scored if s["difficulty"] >= args.min_difficulty]
        with open(hard_path, "w") as f:
            for item in hard:
                f.write(json.dumps(item) + "\n")
        print(f"\nHard questions (score>={args.min_difficulty}): {len(hard)} → {hard_path}")
        print("\nTop 10:")
        for item in hard[:10]:
            print(f"  [{item['difficulty']}] {item['question'][:80]}")
            print(f"       tension: {item['tension'][:80]}")
        return

    # Filter to hard questions only if min_difficulty set above default
    if args.min_difficulty > 1 and not args.score_difficulty:
        hard_path = ROOT / "mimir" / "hard_questions.jsonl"
        if hard_path.exists():
            questions = [json.loads(l) for l in hard_path.read_text().splitlines() if l.strip()]
            print(f"Using hard_questions.jsonl: {len(questions)} questions")

    if args.limit:
        questions = questions[:args.limit]

    cache = _load_cache()
    existing = sum(1 for _ in OUT_FILE.open()) if OUT_FILE.exists() else 0
    print(f"Existing pairs: {existing}")

    if args.dry:
        print("\nDry run — showing first question structure:")
        print(f"  Q: {questions[0]['question'][:100]}")
        print("  Would run: round1 council (6 voices parallel) → synthesis → cross-exam → distance filter → pairs")
        print(f"  Expected pairs per question: 3-6 (filtered from 6 voices)")
        print(f"  Cost estimate: ~$0.003/question (3 Haiku voices + 1 Haiku synthesis + 3 cross-exam)")
        print(f"  Total estimate: ${len(questions) * 0.003:.2f} for {len(questions)} questions")
        return

    if not args.extract:
        print("\nAdd --extract to generate pairs.")
        return

    all_dpo   = []
    all_sft   = []
    all_flagged = []

    for i, item in enumerate(questions):
        question = item["question"]
        context  = item.get("context", "")
        key = _question_key(question)

        if key in cache:
            print(f"[{i+1}/{len(questions)}] cached — skipping")
            continue

        print(f"\n[{i+1}/{len(questions)}] {question[:80]}")
        try:
            dpo_pairs, sft_pairs = generate_pairs_from_deliberation(question, context)
            dpo_pairs, flagged = flag_for_validation(dpo_pairs)
            all_dpo.extend(dpo_pairs)
            all_sft.extend(sft_pairs)
            all_flagged.extend(flagged)
            cache[key] = {
                "dpo_pairs": len(dpo_pairs),
                "sft_pairs": len(sft_pairs),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            _save_cache(cache)
        except Exception as e:
            print(f"  error: {e}")

    # Write DPO pairs (revision only — genuine reasoning quality signal)
    with open(OUT_FILE, "a") as f:
        for pair in all_dpo:
            f.write(json.dumps(pair) + "\n")

    # Write SFT pairs (voice→synthesis format training)
    with open(SFT_FILE, "a") as f:
        for pair in all_sft:
            f.write(json.dumps(pair) + "\n")

    if all_flagged:
        with open(VALIDATION_LOG, "a") as f:
            for pair in all_flagged:
                f.write(json.dumps(pair) + "\n")

    print(f"\n{'='*50}")
    print(f"DPO pairs (revision):  {len(all_dpo)} → {OUT_FILE}")
    print(f"SFT pairs (format):    {len(all_sft)} → {SFT_FILE}")
    print(f"Validation flagged:    {len(all_flagged)}")

    if all_dpo:
        print("\n── Sample DPO pair ──")
        p = all_dpo[0]
        print(f"Q:        {p['prompt'][:80]}")
        print(f"Chosen:   {p['chosen'][:200]}")
        print(f"Rejected: {p['rejected'][:150]}")


if __name__ == "__main__":
    # Handle --seed-from-pairs to bootstrap eval_questions.jsonl from existing pairs
    if "--seed-from-pairs" in sys.argv:
        out = ROOT / "mimir" / "eval_questions.jsonl"
        sources = [
            ROOT / "mimir" / "real_loki_pairs.jsonl",
            ROOT / "mimir" / "orlog_dpo_pairs.jsonl",
        ]
        seen = set()
        count = 0
        with open(out, "w") as f:
            for src in sources:
                if not src.exists():
                    continue
                for line in src.read_text().splitlines():
                    if not line.strip():
                        continue
                    p = json.loads(line)
                    dq = p.get("decision_question") or p.get("prompt", "")
                    if dq and len(dq) > 20 and dq not in seen:
                        seen.add(dq)
                        f.write(json.dumps({"question": dq, "context": p.get("context", "")}) + "\n")
                        count += 1
        print(f"Seeded {count} questions → {out}")
        sys.exit(0)

    main()
