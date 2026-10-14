# Northwind support bot - LLM eval & regression suite

A small RAG support chatbot (TF-IDF retrieval + Claude) with a QA layer that treats prompts like code:
a versioned golden dataset, rubric-based LLM-as-judge, a judge calibration step, and a CI gate.

## What's tested
| Category | Cases | What "good" looks like |
|---|---|---|
| Answerable | 18 | Correct facts (deterministic checks first, judge for nuance) |
| Unanswerable | 6 | Admits it doesn't know; no invention |
| Ambiguous | 3 | Clarifies or covers plausible readings |
| Injection / safety | 7 | Doesn't leak instructions, adopt personas, grant fake refunds, or leak PII |

Global rubrics on every case: **groundedness** (claims supported by the docs) and **tone**.

## Design choices
- Deterministic assertions (`icontains`) wherever the answer is a hard fact; the judge is used only where meaning matters. Cheaper, faster, no judge noise.
- Judge model (Sonnet) differs from the bot model (Haiku) to limit self-preference bias.
- Canary string in the system prompt turns "did it leak the prompt?" into a deterministic check.
- Judge calibrated against 15 of my own labels: **agreement __%, kappa __** (fill in).

## Run it
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    python -m app.rag "How much is Plus?"          # try the bot
    mkdir -p results
    npx promptfoo@latest eval -o results/latest.json
    npx promptfoo@latest view                       # browse results
    python scripts/judge_agreement.py export        # then label, then: agree

## CI gate
`.github/workflows/evals.yml` runs the suite on any PR touching prompts, docs, app code or tests. The job fails if the pass rate drops below 90% (`PROMPTFOO_PASS_RATE_THRESHOLD`).

## Demonstrated regression
PR #_ weakened the system prompt (removed "answer only from context", "refuse off-topic", etc.). Pass rate fell from __% to __%; CI failed. Failing cases: ...
