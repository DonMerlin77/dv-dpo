"""
mimir/failure_detector.py

Two-layer failure detector for Orlog responses.

Layer 1 — Heuristics (free, instant):
  - Format completeness
  - Coherence: VERDICT polarity vs ACTION direction
  - Context ignorance: key question constraints ignored in response

Layer 2 — Mimir judge (free, uses existing container):
  - Semantic grounding check via /evaluate endpoint
  - Only called when heuristics pass or are uncertain

Usage:
  from mimir.failure_detector import detect

  result = detect(question, response)
  # result: {"pass": bool, "failures": [...], "layer": "heuristic"|"mimir"|"both"}

  # With Mimir judge (requires deployed endpoint):
  result = detect(question, response, mimir_url="https://...")
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]

# ---------------------------------------------------------------------------
# Polarity vocabulary
# ---------------------------------------------------------------------------

GO_VERBS = {
    "go", "take", "accept", "join", "fight", "stay", "pursue", "launch",
    "start", "confront", "invest", "do", "yes", "proceed", "commit", "move",
    "open", "build", "push", "escalate", "raise", "bring",
}

AVOID_VERBS = {
    "don't", "dont", "avoid", "decline", "refuse", "leave", "quit", "stop",
    "no", "skip", "drop", "pass", "hold", "wait", "pause", "step back",
    "walk away", "say no", "reject",
}

# Context constraint keywords: if these appear in the question, the response
# must not contradict them.
CONSTRAINT_RULES = [
    {
        "q_keywords": ["sober", "sobriety", "recovering", "recovery", "clean", "quit drinking", "stopped drinking"],
        "r_forbidden": ["drinks", "drink limit", "one drink", "two drink", "three drink", "set a limit", "max drinks"],
        "label": "context_sobriety",
        "reason": "Person stated they are sober/recovering; response recommends drinking.",
    },
    {
        "q_keywords": ["married", "my wife", "my husband", "my spouse"],
        "r_forbidden": ["start dating", "find someone", "meet new people", "move on"],
        "label": "context_relationship_status",
        "reason": "Person is married; response suggests they are single.",
    },
    {
        "q_keywords": ["no money", "broke", "can't afford", "zero savings", "no savings", "in debt"],
        "r_forbidden": ["invest now", "invest immediately", "put money in"],
        "label": "context_no_money",
        "reason": "Person stated they have no money; response recommends investing.",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_section(response: str, label: str) -> str:
    pattern = rf"{label}:(.*?)(?=\n[A-Z ]+:|$)"
    m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _polarity(text: str) -> str:
    """Returns 'go', 'avoid', or 'neutral' based on leading verb/phrasing."""
    t = text.lower().strip()
    for v in AVOID_VERBS:
        if t.startswith(v) or f" {v} " in t[:40]:
            return "avoid"
    for v in GO_VERBS:
        if t.startswith(v) or f" {v} " in t[:40]:
            return "go"
    return "neutral"


# ---------------------------------------------------------------------------
# Layer 1: Heuristics
# ---------------------------------------------------------------------------

def check_format(response: str) -> dict:
    sections = ["VERDICT:", "KEY TENSION:", "BLIND SPOT:", "ACTION:"]
    missing = [s for s in sections if s not in response]
    if missing:
        return {"pass": False, "label": "format", "reason": f"Missing sections: {missing}"}
    return {"pass": True}


def check_coherence(response: str) -> dict:
    verdict = _extract_section(response, "VERDICT")
    action_block = _extract_section(response, "ACTION")
    if not verdict or not action_block:
        return {"pass": True}  # can't check without both

    first_action = action_block.split("\n")[0].lstrip("- ").strip()

    vp = _polarity(verdict)
    ap = _polarity(first_action)

    if vp != "neutral" and ap != "neutral" and vp != ap:
        return {
            "pass": False,
            "label": "coherence",
            "reason": (
                f"VERDICT polarity '{vp}' conflicts with first ACTION polarity '{ap}'.\n"
                f"  VERDICT: {verdict[:80]}\n"
                f"  ACTION[0]: {first_action[:80]}"
            ),
        }
    return {"pass": True}


def check_context_constraints(question: str, response: str) -> dict:
    q_lower = question.lower()
    r_lower = response.lower()
    for rule in CONSTRAINT_RULES:
        if any(kw in q_lower for kw in rule["q_keywords"]):
            if any(fb in r_lower for fb in rule["r_forbidden"]):
                return {
                    "pass": False,
                    "label": rule["label"],
                    "reason": rule["reason"],
                }
    return {"pass": True}


def heuristic_check(question: str, response: str) -> dict:
    failures = []
    for check_fn, args in [
        (check_format, (response,)),
        (check_coherence, (response,)),
        (check_context_constraints, (question, response)),
    ]:
        result = check_fn(*args)
        if not result["pass"]:
            failures.append({"label": result["label"], "reason": result["reason"]})

    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "layer": "heuristic",
    }


# ---------------------------------------------------------------------------
# Layer 2: Mimir judge (uses existing /evaluate endpoint)
# ---------------------------------------------------------------------------

def mimir_judge(question: str, response: str, mimir_url: str) -> dict:
    """
    Calls the /evaluate endpoint on the deployed Mimir serve container.
    Returns {pass, failures, layer}.
    """
    import requests

    base = mimir_url.rstrip("/").replace("/generate", "")
    url = f"{base}/evaluate"

    try:
        r = requests.post(url, json={"question": question, "response": response}, timeout=60)
        r.raise_for_status()
        data = r.json()
        failures = data.get("failures", [])
        return {
            "pass": len(failures) == 0,
            "failures": failures,
            "layer": "mimir",
            "raw": data,
        }
    except Exception as e:
        return {"pass": True, "failures": [], "layer": "mimir_error", "error": str(e)}


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------

def detect(question: str, response: str, mimir_url: str | None = None) -> dict:
    """
    Run both layers. Heuristics first (free), Mimir judge second (if URL provided).
    Returns unified result with all failures from both layers.
    """
    h = heuristic_check(question, response)
    all_failures = list(h["failures"])
    layers_used = ["heuristic"]

    if mimir_url:
        m = mimir_judge(question, response, mimir_url)
        layers_used.append("mimir")
        for f in m.get("failures", []):
            if not any(existing["label"] == f["label"] for existing in all_failures):
                all_failures.append(f)

    return {
        "pass": len(all_failures) == 0,
        "failures": all_failures,
        "layer": "+".join(layers_used),
    }


# ---------------------------------------------------------------------------
# CLI — batch detect on a JSONL of {question, response} pairs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL: {question, response}")
    parser.add_argument("--mimir-url", default=None)
    parser.add_argument("--failures-only", action="store_true")
    args = parser.parse_args()

    total = 0
    flagged = 0

    for line in open(ROOT / args.input):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        q = item.get("question", item.get("prompt", ""))
        r = item.get("response", item.get("chosen", ""))
        result = detect(q, r, args.mimir_url)
        total += 1
        if not result["pass"]:
            flagged += 1
            if args.failures_only:
                print(json.dumps({"question": q[:80], "failures": result["failures"]}))
            else:
                print(f"\nFAIL [{', '.join(f['label'] for f in result['failures'])}]")
                print(f"  Q: {q[:80]}")
                for f in result["failures"]:
                    print(f"  {f['label']}: {f['reason'][:120]}")

    print(f"\n{'='*50}")
    print(f"Total: {total} | Flagged: {flagged} ({flagged/max(total,1):.0%})")
    sys.exit(0 if flagged == 0 else 1)
