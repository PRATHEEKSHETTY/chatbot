"""Minimal RAG support bot: TF-IDF retrieval (one chunk per line, prefixed by its section heading) + one Gemini call."""
import os
import pathlib
import re
import sys
import time

from google import genai
from google.genai import errors, types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_MODEL = os.getenv("APP_MODEL", "gemini-3.1-flash-lite")
TOP_K = int(os.getenv("TOP_K", "5"))
PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "prompts/system.txt")

_STOP = {"the", "is", "a", "an", "of", "to", "do", "i", "my", "how", "what", "can",
         "on", "in", "for", "and", "or", "it", "are", "does", "much", "many", "me", "you"}


def _tok(text):
    words = re.findall(r"[a-z0-9]{2,}", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOP]


def build_chunks(text: str) -> list[str]:
    """One chunk per non-empty line, each prefixed with its '## Heading'.
    Small chunks stop long sections from being diluted in TF-IDF scoring."""
    chunks = []
    for section in re.split(r"\n(?=## )", text):
        section = section.strip()
        if not section.startswith("## "):
            continue
        heading, *lines = section.splitlines()
        chunks += [f"{heading}\n{ln.strip()}" for ln in lines if ln.strip()]
    return chunks


CHUNKS = build_chunks((ROOT / "docs" / "help_center.md").read_text())
_vec = TfidfVectorizer(tokenizer=_tok, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
_matrix = _vec.fit_transform(CHUNKS)
_client = None


def retrieve(question: str, k: int = TOP_K) -> list[str]:
    sims = cosine_similarity(_vec.transform([question]), _matrix)[0]
    top = sims.argsort()[::-1][:k]
    return [CHUNKS[i] for i in top if sims[i] > 0]


def answer(question: str) -> dict:
    global _client
    _client = _client or genai.Client()  # reads GOOGLE_API_KEY
    contexts = retrieve(question)
    system = (ROOT / PROMPT_FILE).read_text().replace(
        "{context}", "\n\n".join(contexts) or "(no relevant excerpts found)"
    )
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=0, max_output_tokens=1024
    )
    for attempt in range(5):
        try:
            resp = _client.models.generate_content(model=APP_MODEL, contents=question, config=config)
            break
        except errors.APIError as e:
            if e.code == 429 and attempt < 4:  # free-tier rate limit: back off and retry
                time.sleep(15 * (attempt + 1))
                continue
            raise
    return {"answer": resp.text or "(empty response)", "contexts": contexts}


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or "How much is the Plus plan?")["answer"])