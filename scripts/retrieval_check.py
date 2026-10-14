"""Retrieval-only check (no LLM calls, free, instant).
For each answerable question, is the right help-center section in the top-k chunks?
Run: python scripts/retrieval_check.py
Separates 'retrieval failed' from 'the model failed' when an eval case breaks."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.rag import retrieve  # noqa: E402

EXPECT = {
    "How much is the Plus plan?": "Plans and pricing",
    "How much storage do I get on the free plan?": "Plans and pricing",
    "What's the biggest single file I can upload on Free?": "File size limits",
    "How long do deleted files stay in Trash on Pro?": "Deleted files",
    "How many days do I have to ask for a refund?": "Refunds",
    "When can I use live chat?": "Support",
    "Can I get my 2FA codes by text message?": "Security",
    "How long is my password reset link valid?": "Security",
    "What's the minimum number of users for the Team plan?": "Plans and pricing",
    "What is the API rate limit?": "API",
    "How far back does version history go on Plus?": "Deleted files",
    "Is my data encrypted?": "Security",
    "Will the desktop app run on Ubuntu 20.04?": "Apps and platforms",
    "Is there a discount for paying yearly?": "Plans and pricing",
    "If I cancel my plan today, do I lose access right away?": "Cancelling",
    "I'm on Free. Can I password-protect a share link?": "Sharing links",
    "We have 5 people. What would the Team plan cost per month?": "Plans and pricing",
    "I signed up in the US region. Can I move my account to the EU?": "Data regions",
}

hits = 0
for question, section in EXPECT.items():
    ok = any(section in c.splitlines()[0] for c in retrieve(question))
    hits += ok
    if not ok:
        print(f"MISS  [{section}]  {question}")
print(f"\nretrieval recall@k = {hits}/{len(EXPECT)}")
sys.exit(0 if hits == len(EXPECT) else 1)