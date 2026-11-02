"""
Step 2 - Clean the raw data, remove duplicates, encode labels, split 70/15/15.

Run:  python scripts/02_clean_and_split.py
In :  data/raw/feedback_raw.csv
Out:  data/processed/{cleaned,train,val,test}.csv, data/labels.csv,
      data/manual_review_sheet.csv, outputs/cleaning_report.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.utils import clean_text, URL_RE, PHONE_RE  # noqa: E402

NEAR_DUP_THRESHOLD = 0.985
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]          # ids 0,1,2
PRIORITY_ORDER = ["Low", "Medium", "High", "Urgent"]           # ids 0,1,2,3
CATEGORY_ORDER = ["Roads / Infrastructure", "Water Supply", "Electricity", "Sanitation",
                  "Health", "Education", "Traffic", "General"]  # ids 0..7

log = []


def note(msg):
    print(msg)
    log.append(msg)


raw = pd.read_csv(ROOT / "data/raw/feedback_raw.csv")
# OPTIONAL: add your own hand-written samples. Put CSV files named manual_*.csv in data/raw/
# with columns: text,category,sentiment,priority  (language/label_source are filled in here).
for extra in sorted((ROOT / "data/raw").glob("manual_*.csv")):
    m = pd.read_csv(extra)
    m["language"] = m.get("language", "manual")
    m["label_source"] = "manual"
    m["id"] = range(raw["id"].max() + 1, raw["id"].max() + 1 + len(m))
    raw = pd.concat([raw, m], ignore_index=True)
    print(f"Added {len(m)} manual samples from {extra.name}")
note(f"Raw rows: {len(raw)}")
note(f"Rows containing URLs: {raw['text'].fillna('').apply(lambda t: bool(URL_RE.search(t))).sum()}")
note(f"Rows containing phone numbers: {raw['text'].fillna('').apply(lambda t: bool(PHONE_RE.search(t))).sum()}")

# 1. missing / empty
df = raw.copy()
df["text"] = df["text"].fillna("")
n0 = len(df)
df = df[df["text"].str.strip() != ""]
note(f"Removed {n0 - len(df)} missing/empty rows")

# 2. clean text (lowercase, URLs, phones, emoji, punctuation, spaces, normalisation)
df["text_clean"] = df["text"].apply(clean_text)
df = df[df["text_clean"].str.len() > 3]

# 3. exact duplicates (on cleaned text)
n0 = len(df)
df = df.drop_duplicates(subset="text_clean", keep="first")
note(f"Removed {n0 - len(df)} exact duplicates")

# 4. near duplicates (cosine similarity on char n-grams, same category)
n0 = len(df)
df = df.reset_index(drop=True)
keep = np.ones(len(df), dtype=bool)
for cat, g in df.groupby("category"):
    idx = g.index.values
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(g["text_clean"])
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    for i in range(len(idx)):
        if not keep[idx[i]]:
            continue
        for j in range(i + 1, len(idx)):
            if keep[idx[j]] and sim[i, j] >= NEAR_DUP_THRESHOLD:
                keep[idx[j]] = False
df = df[keep].reset_index(drop=True)
note(f"Removed {n0 - len(df)} near-duplicates (cosine >= {NEAR_DUP_THRESHOLD})")

# 5. label encoding (explicit, documented mappings)
df["category_id"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
df["sentiment_id"] = df["sentiment"].map({c: i for i, c in enumerate(SENTIMENT_ORDER)})
df["priority_id"] = df["priority"].map({c: i for i, c in enumerate(PRIORITY_ORDER)})
assert df[["category_id", "sentiment_id", "priority_id"]].notna().all().all()
df[["category_id", "sentiment_id", "priority_id"]] = df[["category_id", "sentiment_id", "priority_id"]].astype(int)

# 6. stratified 70 / 15 / 15 split (stratify on category)
train, temp = train_test_split(df, test_size=0.30, stratify=df["category"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["category"], random_state=42)
train = train.assign(split="train"); val = val.assign(split="val"); test = test.assign(split="test")
final = pd.concat([train, val, test]).sort_values("id")

proc = ROOT / "data/processed"
cols = ["id", "text", "text_clean", "category", "category_id", "sentiment", "sentiment_id",
        "priority", "priority_id", "language", "label_source", "split"]
final[cols].to_csv(proc / "cleaned.csv", index=False)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[cols].to_csv(proc / f"{name}.csv", index=False)
final[["id", "text_clean", "category", "category_id", "sentiment", "sentiment_id",
       "priority", "priority_id", "split"]].to_csv(ROOT / "data/labels.csv", index=False)

# 7. sheet for MANUAL review (brief requires >= 200 manually labeled samples)
sheet = final.sample(220, random_state=1)[["id", "text", "category", "sentiment", "priority"]].copy()
sheet.columns = ["id", "text", "auto_category", "auto_sentiment", "auto_priority"]
sheet["verified_category"] = ""
sheet["verified_sentiment"] = ""
sheet["verified_priority"] = ""
sheet.to_csv(ROOT / "data/manual_review_sheet.csv", index=False)

note(f"\nFinal samples: {len(final)} | train {len(train)} / val {len(val)} / test {len(test)}")
note("\nCategory distribution:\n" + final["category"].value_counts().to_string())
note("\nSentiment distribution:\n" + final["sentiment"].value_counts().to_string())
note("\nPriority distribution:\n" + final["priority"].value_counts().to_string())
note("\nLanguage distribution:\n" + final["language"].value_counts().to_string())
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs/cleaning_report.txt").write_text("\n".join(log))
