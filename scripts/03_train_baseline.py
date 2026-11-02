"""
Step 3 - Train + compare baseline models (TF-IDF + sklearn), evaluate, error analysis.

Run:  python scripts/03_train_baseline.py
Out:  models/*.pkl, outputs/{classification_report.txt, confusion_matrix.png,
      model_comparison.csv, sample_predictions.csv, error_analysis.csv}
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.utils import extract_keywords  # noqa: E402

P, M, O = ROOT / "data/processed", ROOT / "models", ROOT / "outputs"
train, val, test = (pd.read_csv(P / f"{n}.csv") for n in ("train", "val", "test"))


def features():
    # word n-grams + character n-grams: char n-grams cope with Roman Urdu spelling variety
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=8000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=15000)),
    ])


def candidates():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=5, class_weight="balanced"),
        "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(C=1, class_weight="balanced"), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    }


def fit_task(task, label_col, save_names):
    """Train all candidates on train, pick best on VALIDATION macro-F1, report on TEST."""
    le = LabelEncoder().fit(pd.concat([train, val, test])[label_col])
    ytr, yva, yte = (le.transform(d[label_col]) for d in (train, val, test))
    rows, fitted = [], {}
    for name, clf in candidates().items():
        pipe = Pipeline([("features", features()), ("clf", clf)]).fit(train["text_clean"], ytr)
        fitted[name] = pipe
        pv, pt = pipe.predict(val["text_clean"]), pipe.predict(test["text_clean"])
        rows.append({"task": task, "model": name,
                     "val_accuracy": accuracy_score(yva, pv), "val_macro_f1": f1_score(yva, pv, average="macro"),
                     "test_accuracy": accuracy_score(yte, pt), "test_macro_f1": f1_score(yte, pt, average="macro")})
    comp = pd.DataFrame(rows)
    # best = highest validation macro-F1 (rounded to 2 decimals); ties -> prefer simpler, probability-friendly models
    pref = {"Logistic Regression": 0, "Linear SVM (calibrated)": 1, "Naive Bayes": 2, "Random Forest": 3}
    comp["_r"] = comp["val_macro_f1"].round(2); comp["_p"] = comp["model"].map(pref)
    best = comp.sort_values(["_r", "_p"], ascending=[False, True]).iloc[0]["model"]
    comp = comp.drop(columns=["_r", "_p"])
    # refit best on train+val for the final saved model? -> NO: keep test untouched & honest.
    model = fitted[best]
    joblib.dump(model, M / save_names[0])
    joblib.dump(le, M / save_names[1])
    return le, model, best, comp, (yte, model.predict(test["text_clean"]))


# ---------------- CATEGORY (main task) ----------------
le_c, m_c, best_c, comp_c, (yte, pte) = fit_task("category", "category", ("baseline_model.pkl", "label_encoder.pkl"))
# ---------------- SENTIMENT + PRIORITY (secondary) ----------------
le_s, m_s, best_s, comp_s, (yte_s, pte_s) = fit_task("sentiment", "sentiment", ("sentiment_model.pkl", "sentiment_label_encoder.pkl"))
le_p, m_p, best_p, comp_p, (yte_p, pte_p) = fit_task("priority", "priority", ("priority_model.pkl", "priority_label_encoder.pkl"))

comp = pd.concat([comp_c, comp_s, comp_p]).round(4)
comp.to_csv(O / "model_comparison.csv", index=False)
print(comp.to_string(index=False))
print(f"\nBest models (by validation macro-F1): category={best_c} | sentiment={best_s} | priority={best_p}")

# keyword TF-IDF vectorizer (unigrams) used by app.utils.extract_keywords
kwv = TfidfVectorizer(ngram_range=(1, 1)).fit(pd.concat([train, val, test])["text_clean"])
joblib.dump(kwv, M / "keyword_vectorizer.pkl")

# ---------------- Reports ----------------
report_lines = []
for title, le, y, p in [("CATEGORY", le_c, yte, pte), ("SENTIMENT", le_s, yte_s, pte_s), ("PRIORITY", le_p, yte_p, pte_p)]:
    report_lines.append(f"===== {title} (test set, n={len(y)}) =====")
    report_lines.append(f"Accuracy: {accuracy_score(y, p):.4f}")
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    report_lines.append(f"Macro precision: {pr:.4f} | Macro recall: {rc:.4f} | Macro F1: {f1:.4f}\n")
    report_lines.append(classification_report(y, p, target_names=le.classes_, zero_division=0))
report_lines.append(f"Selected models: category={best_c}; sentiment={best_s}; priority={best_p}")
(O / "classification_report.txt").write_text("\n".join(report_lines))

# confusion matrices
fig, ax = plt.subplots(figsize=(9, 7.5))
cm = confusion_matrix(yte, pte)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le_c.classes_, yticklabels=le_c.classes_, ax=ax)
ax.set_title(f"Category confusion matrix - {best_c} (test set)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(O / "confusion_matrix.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (title, le, y, p) in zip(axes, [("Sentiment", le_s, yte_s, pte_s), ("Priority", le_p, yte_p, pte_p)]):
    sns.heatmap(confusion_matrix(y, p), annot=True, fmt="d", cmap="Greens", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"{title} confusion matrix (test)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(O / "confusion_matrix_sentiment_priority.png", dpi=150); plt.close()

# class distribution chart
fig, ax = plt.subplots(figsize=(8, 4))
pd.concat([train, val, test])["category"].value_counts().plot.barh(ax=ax, color="#3b82f6")
ax.set_title("Category distribution (all samples)"); plt.tight_layout()
plt.savefig(O / "category_distribution.png", dpi=150); plt.close()

# ---------------- Sample predictions (all test rows) ----------------
proba = m_c.predict_proba(test["text_clean"])
preds = pd.DataFrame({
    "text": test["text"].values,
    "actual_category": le_c.inverse_transform(yte),
    "predicted_category": le_c.inverse_transform(pte),
    "confidence": proba.max(axis=1).round(3),
    "actual_sentiment": le_s.inverse_transform(yte_s), "predicted_sentiment": le_s.inverse_transform(pte_s),
    "actual_priority": le_p.inverse_transform(yte_p), "predicted_priority": le_p.inverse_transform(pte_p),
    "keywords": [", ".join(extract_keywords(t)) for t in test["text"]],
})
preds["category_correct"] = preds["actual_category"] == preds["predicted_category"]
preds.to_csv(O / "sample_predictions.csv", index=False)

# ---------------- Error analysis (>= 15 weak/incorrect predictions) ----------------
err = preds[~preds["category_correct"]].copy()
weak = preds[preds["category_correct"]].sort_values("confidence").head(max(0, 20 - len(err))).copy()
err["error_type"] = "Wrong category"
weak["error_type"] = "Correct but low confidence"


def reason(r):
    t = r["text"].lower()
    if len(t.split()) <= 6:
        return "Very short text: few discriminative keywords"
    topics = {"road": "Roads", "water": "Water", "pani": "Water", "bijli": "Electricity", "electric": "Electricity",
              "hospital": "Health", "school": "Education", "traffic": "Traffic", "garbage": "Sanitation", "gutter": "Sanitation", "sewer": "Sanitation"}
    found = {v for k, v in topics.items() if k in t}
    if len(found) >= 2:
        return "Multi-issue text mentions " + " + ".join(sorted(found))
    if r["confidence"] < 0.45:
        return "Low model confidence: weak/ambiguous training examples"
    return "Shared vocabulary between categories / mixed-language wording"


ea = pd.concat([err, weak])
ea["possible_reason"] = ea.apply(reason, axis=1)
ea = ea.rename(columns={"text": "text_sample"})[
    ["text_sample", "actual_category", "predicted_category", "confidence", "error_type", "possible_reason"]]
ea.to_csv(O / "error_analysis.csv", index=False)
print(f"\nTest accuracy (category): {accuracy_score(yte, pte):.4f}  |  errors: {len(err)}  |  error-analysis rows: {len(ea)}")
print((O / "classification_report.txt").read_text()[:1800])
