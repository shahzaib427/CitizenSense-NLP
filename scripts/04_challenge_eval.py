"""
Step 4 - Honest generalisation check on a HAND-WRITTEN challenge set
(data/raw/challenge_set.csv, ~48 free-form messages that do NOT come from the
template generator). This is the number to quote in your report as
"realistic" performance; the template test-set score is optimistic.

Run:  python scripts/04_challenge_eval.py
Out:  outputs/challenge_results.csv, outputs/challenge_report.txt
      (+ appends wrong predictions to outputs/error_analysis.csv)
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.utils import analyze_feedback  # noqa: E402

df = pd.read_csv(ROOT / "data/raw/challenge_set.csv")
res = [analyze_feedback(t) for t in df["text"]]
df["pred_category"] = [r["category"] for r in res]
df["pred_sentiment"] = [r["sentiment"] for r in res]
df["pred_priority"] = [r["priority"] for r in res]
df["confidence"] = [r["confidence"]["category"] for r in res]
df["keywords"] = [", ".join(r["keywords"]) for r in res]
df["category_correct"] = df["category"] == df["pred_category"]
df.to_csv(ROOT / "outputs/challenge_results.csv", index=False)

lines = [f"Challenge set size: {len(df)}"]
for name, a, b in [("category", "category", "pred_category"), ("sentiment", "sentiment", "pred_sentiment"),
                   ("priority", "priority", "pred_priority")]:
    lines.append(f"\n=== {name.upper()} === accuracy {accuracy_score(df[a], df[b]):.3f} | macro-F1 {f1_score(df[a], df[b], average='macro'):.3f}")
    if name == "category":
        lines.append(classification_report(df[a], df[b], zero_division=0))
report = "\n".join(lines)
(ROOT / "outputs/challenge_report.txt").write_text(report)
print(report)

# append wrong predictions to error analysis
wrong = df[~df["category_correct"]].copy()
wrong = wrong.rename(columns={"text": "text_sample", "category": "actual_category", "pred_category": "predicted_category"})
wrong["error_type"] = "Wrong category (challenge set)"
wrong["possible_reason"] = "REVIEW MANUALLY: short / multi-issue / unseen wording"
ea_path = ROOT / "outputs/error_analysis.csv"
ea = pd.read_csv(ea_path)
ea = ea[~ea["error_type"].str.contains("challenge", na=False)]
cols = ["text_sample", "actual_category", "predicted_category", "confidence", "error_type", "possible_reason"]
pd.concat([ea, wrong[cols]]).to_csv(ea_path, index=False)
print(f"\nAppended {len(wrong)} challenge errors -> error_analysis.csv now has {len(ea) + len(wrong)} rows")
