"""
Shared logic for CitizenSense: text cleaning, keyword extraction, summary,
department routing, and the end-to-end `analyze_feedback()` function used by
the CLI, Streamlit app, FastAPI API and the notebook.

Everything here works OFFLINE with the scikit-learn models in /models.
Hugging Face models are used only when you call the optional helpers
(`hf_sentiment`, `hf_summarize`, `hf_zero_shot`) - see app/hf_models.py.
"""
import re
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# --------------------------------------------------------------------------
# 1. Text cleaning
# --------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|org|net|pk|ly)/\S*)", re.I)
PHONE_RE = re.compile(r"(\+?92[\s-]?|0)3\d{2}[\s-]?\d{7}\b|\b\d{11}\b")
CNIC_RE = re.compile(r"\b\d{5}-?\d{7}-?\d\b")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]+", flags=re.UNICODE)

# small, safe spelling normalisation (typos + common Roman Urdu variants)
NORMALIZE = {
    "electrcity": "electricity", "garbge": "garbage", "hosptial": "hospital",
    "watter": "water", "scool": "school", "trafic": "traffic", "rode": "road",
    "plz": "please", "urgentlyy": "urgently",
    "sadak": "sarak", "sarkar": "sarak", "pani": "pani", "paani": "pani",
    "bijlee": "bijli", "bjli": "bijli", "gutar": "gutter", "kachra": "kooda",
}


def clean_text(text) -> str:
    """Lowercase, strip URLs / phones / CNIC / emails / emoji, fix spaces."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = str(text)
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = CNIC_RE.sub(" ", t)
    t = PHONE_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = t.lower()
    t = re.sub(r"[^\w\s\u0600-\u06FF']", " ", t)  # keep letters, digits, Urdu script
    t = re.sub(r"\s+", " ", t).strip()
    t = " ".join(NORMALIZE.get(w, w) for w in t.split())
    return t


# --------------------------------------------------------------------------
# 2. Keyword extraction (TF-IDF scores + domain lexicon boost)
# --------------------------------------------------------------------------
STOPWORDS = set("""
a an the and or but if of to in on at for from with by is are was were be been being it its this that these those
i we you he she they our your their my me us them there here has have had do does did not no than then so very
too just also can could will would should may might please kindly sir dear officer respected hello complaint
want report area locality neighbourhood colony block near every day since
hai hain tha thi thay ho hota hoti hotay ka ki ke ko se mein me par pe aur lekin ya bhi to hi nahi nahin
hamare hamari hamara humein hum mera meri mere yeh ye woh wo is us un kya kab kahan kaise kis kuch koi
raha rahi rahe gaya gayi gaye kar karein karke karo baraye barae meherbani janab assalam alaikum
bohat bahut zyada kam bilkul abhi phir kai ek do teen din mahine hafte
one two three four five six seven eight nine ten day days week weeks month months hour hours yesterday monday
much now thank thanks shukriya better get how will someone act new outside
""".split())

DOMAIN_TERMS = {
    "road", "sarak", "pothole", "potholes", "gaddhe", "bridge", "pul", "manhole", "footpath", "construction",
    "water", "pani", "pipeline", "tanker", "tap", "supply", "contaminated", "leak", "leaking", "tube", "well",
    "electricity", "bijli", "transformer", "wires", "taarein", "load", "shedding", "voltage", "meter", "bill", "pole",
    "garbage", "kooda", "sewer", "gutter", "naali", "drain", "waste", "dustbin", "mosquitoes", "machar", "sewage",
    "hospital", "doctor", "medicines", "dawaiyan", "ambulance", "oxygen", "dengue", "patients", "mareez", "staff",
    "school", "teachers", "ustad", "students", "books", "kitabein", "classrooms", "scholarship", "wazifa",
    "traffic", "signal", "jam", "parking", "accident", "accidents", "hadsay", "warden", "intersection", "chowk",
    "rainwater", "barish", "dengue", "dogs", "kutte", "park", "noise", "encroached",
}

_kw_vectorizer = None


def _get_kw_vectorizer():
    global _kw_vectorizer
    if _kw_vectorizer is None:
        path = MODELS_DIR / "keyword_vectorizer.pkl"
        _kw_vectorizer = joblib.load(path) if path.exists() else False
    return _kw_vectorizer


def extract_keywords(text: str, top_k: int = 4):
    """Return the top-k keywords/phrases. Works with or without the fitted
    TF-IDF vectorizer (falls back to frequency + domain lexicon)."""
    cleaned = clean_text(text)
    tokens = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]
    if not tokens:
        return []
    vec = _get_kw_vectorizer()
    idf = {}
    if vec:
        idf = dict(zip(vec.get_feature_names_out(), vec.idf_))

    scores = {}
    for i, w in enumerate(tokens):
        s = idf.get(w, 1.0) + (1.5 if w in DOMAIN_TERMS else 0.0)
        scores[w] = scores.get(w, 0) + s
    # bigram phrases made of adjacent non-stopwords that contain a domain term
    raw_words = cleaned.split()
    for a, b in zip(raw_words, raw_words[1:]):
        if a in STOPWORDS or b in STOPWORDS or len(a) < 3 or len(b) < 3:
            continue
        if a in DOMAIN_TERMS or b in DOMAIN_TERMS:
            scores[f"{a} {b}"] = idf.get(a, 1) + idf.get(b, 1) + 1.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    chosen = []
    for kw, _ in ranked:
        # avoid listing a unigram that is already inside a chosen bigram
        if any(kw in c.split() for c in chosen if " " in c):
            continue
        chosen.append(kw)
        if len(chosen) == top_k:
            break
    return chosen


# --------------------------------------------------------------------------
# 3. Department routing, action and summary
# --------------------------------------------------------------------------
DEPARTMENTS = {
    "Roads / Infrastructure": ("Municipal / Infrastructure (Roads & Works)", "Inspect and repair the damaged road or structure"),
    "Water Supply": ("Water & Sanitation Agency (WASA)", "Inspect supply line, fix leakage and restore clean water"),
    "Electricity": ("Electricity Distribution Company (DISCO)", "Dispatch a technical team to restore safe power"),
    "Sanitation": ("Municipal Sanitation / Solid Waste Management", "Schedule cleaning, waste lifting or sewer clearing"),
    "Health": ("Health Department / Hospital Administration", "Improve patient services, staffing and medicine availability"),
    "Education": ("School Education Department", "Review school facilities, staffing and student services"),
    "Traffic": ("Traffic Police / Transport Department", "Fix signals, enforce rules and improve traffic management"),
    "General": ("Citizen Services / Complaint Cell", "Route to the relevant office and update the citizen"),
}
CATEGORY_PHRASE = {
    "Roads / Infrastructure": "a road or infrastructure problem",
    "Water Supply": "a water supply problem",
    "Electricity": "an electricity problem",
    "Sanitation": "a sanitation problem",
    "Health": "a health service problem",
    "Education": "an education problem",
    "Traffic": "a traffic problem",
    "General": "a general civic issue",
}
DURATION_RE = re.compile(
    r"(since yesterday|for (?:\w+ ){0,2}(?:days?|weeks?|months?|hours?)|"
    r"(?:kal|do din|teen din|ek hafte|kai hafton|kai mahine|kai dinon|kai ghantay) se|every day|rozana)")
URGENT_RE = re.compile(r"urgent|emergency|immediately|asap|jaldi|fori|critical|dangerous|khatarnak|khatra")


def summarize(text: str, category: str, sentiment: str, keywords) -> str:
    """Template-based extractive summary (offline, deterministic)."""
    low = text.lower()
    dur = DURATION_RE.search(low)
    dur_txt = f" ({dur.group(1)})" if dur else ""
    kw = ", ".join(keywords[:3]) if keywords else "the reported issue"
    if sentiment == "Positive":
        return f"Citizen appreciates the service related to {kw}."
    if sentiment == "Neutral":
        return f"Citizen asks for information about {kw}."
    ask = "requests urgent action" if URGENT_RE.search(low) else "requests action"
    return f"Citizen reports {CATEGORY_PHRASE.get(category, 'an issue')} involving {kw}{dur_txt} and {ask}."


# --------------------------------------------------------------------------
# 4. Model loading + the main analysis function
# --------------------------------------------------------------------------
_models = {}
REVIEW_THRESHOLD = 0.45  # below this, send to a human reviewer


def load_models():
    if not _models:
        _models["category"] = joblib.load(MODELS_DIR / "baseline_model.pkl")
        _models["category_le"] = joblib.load(MODELS_DIR / "label_encoder.pkl")
        _models["sentiment"] = joblib.load(MODELS_DIR / "sentiment_model.pkl")
        _models["sentiment_le"] = joblib.load(MODELS_DIR / "sentiment_label_encoder.pkl")
        _models["priority"] = joblib.load(MODELS_DIR / "priority_model.pkl")
        _models["priority_le"] = joblib.load(MODELS_DIR / "priority_label_encoder.pkl")
    return _models


def _predict(task: str, cleaned: str):
    m = load_models()
    model, le = m[task], m[f"{task}_le"]
    proba = model.predict_proba([cleaned])[0]
    idx = int(proba.argmax())
    return le.inverse_transform([idx])[0], float(proba[idx]), {c: float(p) for c, p in zip(le.classes_, proba)}


def analyze_feedback(text: str) -> dict:
    """Full CitizenSense insight card for one feedback message."""
    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "Empty text after cleaning. Please enter some feedback."}
    category, c_conf, c_probs = _predict("category", cleaned)
    sentiment, s_conf, _ = _predict("sentiment", cleaned)
    priority, p_conf, _ = _predict("priority", cleaned)
    # safety rule: danger words always escalate to at least High
    if URGENT_RE.search(text.lower()) and priority in ("Low", "Medium"):
        priority = "High"
    keywords = extract_keywords(text)
    dept, action = DEPARTMENTS[category]
    return {
        "category": category,
        "sentiment": sentiment,
        "priority": priority,
        "keywords": keywords,
        "suggested_department": dept,
        "suggested_action": action,
        "summary": summarize(text, category, sentiment, keywords),
        "confidence": {"category": round(c_conf, 3), "sentiment": round(s_conf, 3), "priority": round(p_conf, 3)},
        "category_probabilities": {k: round(v, 3) for k, v in sorted(c_probs.items(), key=lambda kv: -kv[1])},
        "needs_human_review": bool(c_conf < REVIEW_THRESHOLD),
        "cleaned_text": cleaned,
    }
