# DV-DPO: Deliberation-Validated DPO

A training pipeline that generates its own DPO pairs from AI deliberation — no human labelers required.

## The idea

Standard DPO needs (chosen, rejected) pairs. Getting good ones is expensive. This pipeline generates them automatically by running a multi-voice council on each question and only capturing pairs when the synthesis gets revised under cross-examination.

If the synthesis holds up — no pair. If it gets revised — that revision *is* the training signal.

## How it works

1. Three council voices deliberate on a question from asymmetric starting positions
2. A synthesis is produced
3. Losing voices cross-examine the synthesis
4. If the synthesis is **revised** → DPO pair (chosen=post-revision, rejected=pre-revision)
5. If the synthesis is **upheld** → no pair (good reasoning, nothing to improve)

Only genuine revisions under adversarial pressure become training data. Not format preference, not sampling variance.

## Results

Fine-tuned Qwen2.5-7B on 1,040 pairs generated this way (~$3 at Haiku API rates):

| Model | Format | Commits | Context | Composite | Latency |
|-------|--------|---------|---------|-----------|---------|
| Fine-tuned 7B | 100% | 100% | 89% | 96% | 11.0s |
| Claude Haiku | 100% | 100% | 100% | 100% | 3.0s |

Adversarial failure rate: 2% on 96 targeted questions.

## Files

- `deliberation_dpo.py` — core pipeline (council → synthesis → cross-exam → pairs)
- `failure_detector.py` — heuristic quality checker for responses
- `auto_red_team.py` — detect failures, generate corrected pairs automatically
- `auto_train_loop.py` — full cycle: detect → pair → train → deploy → eval
- `eval_questions.jsonl` — 1,044 seed questions
- `adversarial_questions.jsonl` — 96 adversarial questions targeting known failure modes

## Setup

```bash
pip install anthropic python-dotenv
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

## Run

```bash
# Generate DPO pairs from seed questions
python deliberation_dpo.py

# Run adversarial red-team
python auto_red_team.py

# Full autonomous loop
python auto_train_loop.py
```

## Quality metric

Revision rate over time. As the model improves, easy questions stop producing revisions — harder questions are needed. A dropping revision rate is measurable improvement without a separate benchmark.
