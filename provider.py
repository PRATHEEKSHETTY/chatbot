"""Promptfoo custom provider: runs the RAG bot for each test question."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from app.rag import answer  # noqa: E402


def call_api(prompt, options, context):
    question = context["vars"]["question"]
    result = answer(question)
    return {"output": result["answer"], "metadata": {"contexts": result["contexts"]}}
