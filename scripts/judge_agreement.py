"""Calibrate the LLM judge against your own labels.

1) python scripts/judge_agreement.py export     -> labels/label_sheet.csv (15 stratified cases, judge verdict hidden)
2) Fill in the human_pass column (1 = acceptable answer, 0 = not), save as labels/human_labels.csv
3) python scripts/judge_agreement.py agree      -> agreement %, Cohen's kappa, disagreements
"""
import csv
import json
import pathlib
import sys
from collections import defaultdict

RESULTS = pathlib.Path("results/latest.json")
SHEET = pathlib.Path("labels/label_sheet.csv")
LABELS = pathlib.Path("labels/human_labels.csv")


def load_rows():
    return json.loads(RESULTS.read_text())["results"]["results"]


def judge_pass(row):
    comps = (row.get("gradingResult") or {}).get("componentResults") or []
    rubric = [c for c in comps if (c.get("assertion") or {}).get("type") == "llm-rubric"]
    return all(c.get("pass") for c in rubric) if rubric else None


def desc(row):
    return row["testCase"]["description"]


def export(n=15):
    by_cat = defaultdict(list)
    for r in load_rows():
        by_cat[(r["testCase"].get("metadata") or {}).get("category", "?")].append(r)
    picked, i = [], 0
    while len(picked) < n and any(by_cat.values()):
        for cat in list(by_cat):
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop(0))
        i += 1
    SHEET.parent.mkdir(exist_ok=True)
    with SHEET.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["description", "question", "answer", "human_pass"])
        for r in picked:
            w.writerow([desc(r), r["testCase"]["vars"]["question"], (r.get("response") or {}).get("output", ""), ""])
    print(f"Wrote {len(picked)} rows to {SHEET}. Fill human_pass (1/0) and save as {LABELS}.")


def agree():
    judge = {desc(r): judge_pass(r) for r in load_rows()}
    pairs = []
    with LABELS.open() as f:
        for row in csv.DictReader(f):
            h = row["human_pass"].strip().lower()
            if h in ("1", "0", "pass", "fail") and judge.get(row["description"]) is not None:
                pairs.append((row["description"], h in ("1", "pass"), judge[row["description"]]))
    n = len(pairs)
    if not n:
        sys.exit("No labelled rows matched.")
    tp = sum(h and j for _, h, j in pairs)
    tn = sum((not h) and (not j) for _, h, j in pairs)
    fp = sum((not h) and j for _, h, j in pairs)  # judge too lenient
    fn = sum(h and (not j) for _, h, j in pairs)  # judge too strict
    po = (tp + tn) / n
    pe = ((tp + fn) / n) * ((tp + fp) / n) + ((tn + fp) / n) * ((tn + fn) / n)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    print(f"n={n}  agreement={po:.0%}  Cohen's kappa={kappa:.2f}")
    print(f"judge too lenient (human fail, judge pass): {fp}   judge too strict (human pass, judge fail): {fn}")
    for d, h, j in pairs:
        if h != j:
            print(f"  DISAGREE  {d}: human={'pass' if h else 'fail'} judge={'pass' if j else 'fail'}")


if __name__ == "__main__":
    {"export": export, "agree": agree}[sys.argv[1] if len(sys.argv) > 1 else "agree"]()
