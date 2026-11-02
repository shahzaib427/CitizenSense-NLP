"""
Hugging Face / transformer components (optional at demo time, REQUIRED for the task).
Needs internet + `pip install transformers torch sentence-transformers`.
Models download on first use (use a Colab GPU or be patient on CPU).

  hf_sentiment()      multilingual sentiment (XLM-R)         -> label
  hf_zero_shot()      zero-shot category (BART-MNLI)         -> label + score
  hf_summarize()      abstractive summary for LONG feedback   -> text
  embedding_classify  multilingual MiniLM embeddings + LogReg -> category
"""
from functools import lru_cache

CATEGORIES = ["Roads / Infrastructure", "Water Supply", "Electricity", "Sanitation",
              "Health", "Education", "Traffic", "General"]
# natural-language hypotheses work better for zero-shot than raw labels
ZS_LABELS = {
    "road, bridge or street damage": "Roads / Infrastructure",
    "water supply or water quality": "Water Supply",
    "electricity or power supply": "Electricity",
    "garbage, sewer or sanitation": "Sanitation",
    "hospital or health service": "Health",
    "school or education": "Education",
    "traffic or road safety": "Traffic",
    "other general civic issue": "General",
}


@lru_cache(maxsize=1)
def _sentiment_pipe():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")


@lru_cache(maxsize=1)
def _zs_pipe():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@lru_cache(maxsize=1)
def _sum_pipe():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def hf_sentiment(text: str) -> str:
    out = _sentiment_pipe()(text[:512])[0]["label"].capitalize()
    return {"Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"}.get(out, out)


def hf_zero_shot(text: str):
    r = _zs_pipe()(text, candidate_labels=list(ZS_LABELS), hypothesis_template="This complaint is about {}.")
    return ZS_LABELS[r["labels"][0]], float(r["scores"][0])


def hf_summarize(text: str, max_len: int = 40) -> str:
    if len(text.split()) < 40:           # short feedback: summarizer would just copy it
        return text
    return _sum_pipe()(text, max_length=max_len, min_length=10, do_sample=False)[0]["summary_text"]


def embed(texts):
    return _embedder().encode(list(texts), show_progress_bar=False, normalize_embeddings=True)


def train_embedding_classifier(train_texts, train_labels):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=10).fit(embed(train_texts), train_labels)
