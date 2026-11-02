"""
CitizenSense - Streamlit dashboard.
Run from the project root:  streamlit run app/streamlit_app.py
"""
import html
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.utils import REVIEW_THRESHOLD, analyze_feedback  # noqa: E402

st.set_page_config(page_title="CitizenSense", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------- design tokens
INK, MUTED, LINE, PAPER, TEAL = "#17212B", "#5B6773", "#DDE3E7", "#F4F6F7", "#0F5C5C"
PRIORITY = {  # colour, soft background, what it means for the officer
    "Low": ("#4B5D6B", "#EAEFF2", "Routine - handle in normal queue"),
    "Medium": ("#9A6700", "#FFF4D6", "Plan a response this week"),
    "High": ("#B4410C", "#FDE9DD", "Assign a team promptly"),
    "Urgent": ("#A11A1A", "#FBE0E0", "Escalate now - possible risk to people"),
}
SENTIMENT = {"Negative": ("#A11A1A", "#FBE0E0"), "Neutral": ("#4B5D6B", "#EAEFF2"), "Positive": ("#146C43", "#DCF3E6")}
PRIORITY_ORDER = ["Urgent", "High", "Medium", "Low"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, textarea, input {{ font-family: 'Public Sans', 'Segoe UI', Arial, sans-serif !important; }}
.stApp {{ background: {PAPER}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

.cs-brand {{ display:flex; align-items:center; gap:.7rem; margin-bottom:.15rem; }}
.cs-mark {{ width:38px; height:38px; border-radius:9px; background:{TEAL}; color:#fff; display:flex;
            align-items:center; justify-content:center; font-weight:700; font-size:1.05rem; }}
.cs-title {{ font-size:1.55rem; font-weight:700; color:{INK}; letter-spacing:-.01em; line-height:1.1; }}
.cs-sub {{ color:{MUTED}; font-size:.92rem; margin:.15rem 0 1.1rem 0; }}

.cs-card {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:1.25rem 1.4rem; margin-bottom:1rem; }}
.cs-result {{ border-left:7px solid var(--pc); }}
.cs-banner {{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; }}
.cs-cat {{ font-size:1.6rem; font-weight:700; color:{INK}; line-height:1.15; }}
.cs-small {{ color:{MUTED}; font-size:.85rem; }}
.cs-pill {{ display:inline-block; padding:.28rem .75rem; border-radius:999px; font-weight:600; font-size:.85rem; margin-right:.4rem; }}
.cs-prio {{ text-align:right; }}
.cs-prio .lvl {{ font-size:1.25rem; font-weight:700; color:var(--pc); }}
.cs-sec {{ font-weight:600; color:{INK}; font-size:.95rem; margin:1rem 0 .4rem 0; }}
.cs-chip {{ display:inline-block; background:{PAPER}; border:1px solid {LINE}; color:{INK}; border-radius:6px;
            padding:.22rem .6rem; margin:0 .35rem .35rem 0; font-size:.88rem; }}
.cs-summary {{ background:{PAPER}; border-radius:8px; padding:.8rem 1rem; color:{INK}; line-height:1.5; }}
.cs-route {{ display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }}
.cs-route div {{ background:{PAPER}; border-radius:8px; padding:.7rem .9rem; }}
.cs-route b {{ display:block; font-size:.8rem; color:{MUTED}; font-weight:500; margin-bottom:.15rem; }}
.cs-meter {{ height:8px; background:{LINE}; border-radius:99px; overflow:hidden; margin-top:.35rem; }}
.cs-meter span {{ display:block; height:100%; background:{TEAL}; border-radius:99px; }}
.cs-review {{ background:#FFF4D6; border:1px solid #E8C766; color:#6B4A00; border-radius:8px; padding:.7rem 1rem; margin-top:1rem; font-size:.92rem; }}
.cs-empty {{ border:1.5px dashed {LINE}; border-radius:10px; padding:2.2rem 1rem; text-align:center; color:{MUTED}; background:#fff; }}
.cs-kpi {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.9rem 1.1rem; }}
.cs-kpi .n {{ font-size:1.8rem; font-weight:700; color:{INK}; line-height:1.1; }}
.cs-kpi .l {{ color:{MUTED}; font-size:.85rem; }}
div.stButton > button {{ border-radius:8px; font-weight:600; }}
div.stButton > button[kind="primary"] {{ background:{TEAL}; border:1px solid {TEAL}; }}
div.stButton > button[kind="primary"]:hover {{ background:#0B4747; border-color:#0B4747; }}
button[data-baseweb="tab"] {{ font-weight:600; }}
textarea {{ border-radius:8px !important; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- state
st.session_state.setdefault("history", [])
st.session_state.setdefault("text", "")
st.session_state.setdefault("result", None)

EXAMPLES = {
    "Water outage (English)": "There is no water supply in our area for three days.",
    "Damaged road (Roman Urdu)": "hamare area ki road bohat kharab hai aur pani jama hota hai",
    "Dangerous wires": "Loose electric wires are hanging near the street, please act immediately!",
    "Hospital waiting time": "The hospital staff was helpful but the waiting time was too long.",
    "Appreciation": "Thank you for repairing the road in our area, it is much better now.",
    "Information request": "How can I apply for a new electricity meter?",
}


def load_example(key):
    st.session_state["text"] = EXAMPLES[key]
    st.session_state["result"] = None


def esc(x):
    return html.escape(str(x))


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="cs-brand"><div class="cs-mark">CS</div><div class="cs-title" style="font-size:1.25rem">CitizenSense</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-small" style="margin-bottom:1rem">Public feedback triage prototype</div>', unsafe_allow_html=True)
    st.markdown("**Try an example**")
    for k in EXAMPLES:
        st.button(k, key=f"ex_{k}", on_click=load_example, args=(k,), width="stretch")
    st.divider()
    st.markdown("**Settings**")
    threshold = st.slider("Send to human review below", 0.20, 0.90, float(REVIEW_THRESHOLD), 0.05,
                          help="Predictions with category confidence under this value are flagged for a person to check.")
    show_probs = st.toggle("Show category probabilities", value=True)
    st.divider()
    st.caption("Learning prototype built on synthetic data. Results are suggestions for a human reader, not decisions. "
               "Do not enter names, phone numbers or CNIC numbers.")


def needs_review(r):
    return r["confidence"]["category"] < threshold


# ----------------------------------------------------------------- header
st.markdown('<div class="cs-brand"><div class="cs-mark">CS</div><div class="cs-title">CitizenSense</div></div>', unsafe_allow_html=True)
st.markdown('<div class="cs-sub">Read a citizen message in English, Urdu or Roman Urdu and get its category, priority, summary and the team that should handle it.</div>', unsafe_allow_html=True)

tab_one, tab_batch, tab_perf = st.tabs(["Analyze a message", "Analyze a file", "Model and limits"])

# ================================================================= TAB 1
with tab_one:
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.text_area("Citizen message", key="text", height=190,
                     placeholder="Paste or type the complaint here, for example: there is no water in our street for three days")
        c1, c2 = st.columns([1, 1])
        run = c1.button("Analyze message", type="primary", width="stretch")
        if c2.button("Clear", width="stretch"):
            st.session_state["text"] = ""
            st.session_state["result"] = None
            st.rerun()

        if run:
            r = analyze_feedback(st.session_state["text"])
            st.session_state["result"] = r
            if "error" not in r:
                st.session_state["history"].insert(0, {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Message": st.session_state["text"][:80] + ("..." if len(st.session_state["text"]) > 80 else ""),
                    "Category": r["category"], "Priority": r["priority"], "Sentiment": r["sentiment"],
                    "Confidence": f'{r["confidence"]["category"]:.0%}'})

        if st.session_state["history"]:
            st.markdown('<div class="cs-sec">Recent analyses</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state["history"][:8]), hide_index=True)

    with right:
        r = st.session_state["result"]
        if r is None:
            st.markdown('<div class="cs-empty"><b>No message analyzed yet</b><br>Type a message or pick an example from the sidebar, then choose Analyze message.</div>', unsafe_allow_html=True)
        elif "error" in r:
            st.error(f"{r['error']}")
        else:
            pc, pbg, meaning = PRIORITY[r["priority"]]
            sc, sbg = SENTIMENT[r["sentiment"]]
            conf = r["confidence"]["category"]
            chips = "".join(f'<span class="cs-chip">{esc(k)}</span>' for k in r["keywords"]) or '<span class="cs-small">No keywords found</span>'
            review = ('<div class="cs-review"><b>Check this one manually.</b> The model is only '
                      f'{conf:.0%} sure of the category, below your {threshold:.0%} limit.</div>') if needs_review(r) else ""
            st.markdown(f"""
<div class="cs-card cs-result" style="--pc:{pc}">
  <div class="cs-banner">
    <div>
      <div class="cs-small">Category</div>
      <div class="cs-cat">{esc(r['category'])}</div>
      <div style="margin-top:.6rem">
        <span class="cs-pill" style="background:{sbg};color:{sc}">{esc(r['sentiment'])} sentiment</span>
      </div>
    </div>
    <div class="cs-prio">
      <div class="cs-small">Priority</div>
      <div class="lvl">{esc(r['priority'])}</div>
      <div class="cs-small" style="max-width:210px">{esc(meaning)}</div>
    </div>
  </div>
  <div class="cs-sec">Category confidence: {conf:.0%}</div>
  <div class="cs-meter"><span style="width:{conf*100:.0f}%"></span></div>
  <div class="cs-sec">Summary</div>
  <div class="cs-summary">{esc(r['summary'])}</div>
  <div class="cs-sec">Key phrases</div>
  <div>{chips}</div>
  <div class="cs-sec">Where it should go</div>
  <div class="cs-route">
    <div><b>Department</b>{esc(r['suggested_department'])}</div>
    <div><b>Suggested action</b>{esc(r['suggested_action'])}</div>
  </div>
  {review}
</div>""", unsafe_allow_html=True)

            if show_probs:
                probs = pd.DataFrame({"Category": list(r["category_probabilities"]), "Probability": list(r["category_probabilities"].values())})
                chart = (alt.Chart(probs).mark_bar(color=TEAL, cornerRadiusEnd=3, size=16)
                         .encode(x=alt.X("Probability:Q", axis=alt.Axis(format="%", title=None, grid=False), scale=alt.Scale(domain=[0, 1])),
                                 y=alt.Y("Category:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220, labelFontSize=12)),
                                 tooltip=["Category", alt.Tooltip("Probability:Q", format=".1%")])
                         .properties(height=280, title="How the model weighed each category")
                         .configure_view(strokeWidth=0).configure_title(anchor="start", fontSize=13, color=INK, font="Public Sans"))
                st.altair_chart(chart, width="stretch")
            with st.expander("Text the model actually read"):
                st.code(r["cleaned_text"], language=None)

# ================================================================= TAB 2
with tab_batch:
    st.markdown("Upload a CSV with a column named **text**. Up to 500 rows are analyzed.")
    up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    sample_path = ROOT / "outputs" / "challenge_results.csv"
    if sample_path.exists():
        st.download_button("Download a sample file to try", pd.read_csv(sample_path)[["text"]].to_csv(index=False),
                           "sample_feedback.csv", "text/csv")
    if up is None:
        st.markdown('<div class="cs-empty" style="margin-top:.8rem"><b>No file yet</b><br>Upload a CSV to see category and priority charts and a list of messages that need review.</div>', unsafe_allow_html=True)
    else:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("This file has no column named 'text'. Rename the message column to 'text' and upload it again.")
        else:
            with st.spinner("Analyzing messages..."):
                rows = []
                for t in df["text"].dropna().astype(str).head(500):
                    r = analyze_feedback(t)
                    if "error" not in r:
                        rows.append({"Message": t, "Category": r["category"], "Sentiment": r["sentiment"], "Priority": r["priority"],
                                     "Confidence": round(r["confidence"]["category"] * 100), "Key phrases": ", ".join(r["keywords"]),
                                     "Department": r["suggested_department"], "Review": r["confidence"]["category"] < threshold})
            out = pd.DataFrame(rows)
            if out.empty:
                st.warning("No readable messages were found in the file.")
            else:
                k = st.columns(4)
                kp = [("Messages analyzed", len(out)), ("Urgent or high priority", int(out["Priority"].isin(["Urgent", "High"]).sum())),
                      ("Negative sentiment", f'{(out["Sentiment"] == "Negative").mean():.0%}'), ("Need human review", int(out["Review"].sum()))]
                for col, (lab, val) in zip(k, kp):
                    col.markdown(f'<div class="cs-kpi"><div class="n">{val}</div><div class="l">{lab}</div></div>', unsafe_allow_html=True)
                st.write("")
                a, b = st.columns(2, gap="large")
                cat = out["Category"].value_counts().rename_axis("Category").reset_index(name="Messages")
                a.altair_chart(alt.Chart(cat).mark_bar(color=TEAL, cornerRadiusEnd=3)
                               .encode(x=alt.X("Messages:Q", title=None, axis=alt.Axis(grid=False)), y=alt.Y("Category:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                                       tooltip=["Category", "Messages"]).properties(height=280, title="Messages by category")
                               .configure_title(anchor="start", fontSize=13, color=INK), width="stretch")
                pr = out["Priority"].value_counts().rename_axis("Priority").reset_index(name="Messages")
                b.altair_chart(alt.Chart(pr).mark_bar(cornerRadiusEnd=3)
                               .encode(x=alt.X("Messages:Q", title=None, axis=alt.Axis(grid=False)), y=alt.Y("Priority:N", sort=PRIORITY_ORDER, title=None, axis=alt.Axis(labelOverlap=False)),
                                       color=alt.Color("Priority:N", sort=PRIORITY_ORDER, legend=None,
                                                       scale=alt.Scale(domain=PRIORITY_ORDER, range=[PRIORITY[p][0] for p in PRIORITY_ORDER])),
                                       tooltip=["Priority", "Messages"]).properties(height=280, title="Messages by priority")
                               .configure_title(anchor="start", fontSize=13, color=INK), width="stretch")

                st.markdown('<div class="cs-sec">Messages to look at first</div>', unsafe_allow_html=True)
                rank = {p: i for i, p in enumerate(PRIORITY_ORDER)}
                queue = out.assign(_r=out["Priority"].map(rank)).sort_values(["_r", "Confidence"]).drop(columns="_r")
                st.dataframe(queue, hide_index=True, column_config={
                    "Confidence": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100),
                    "Review": st.column_config.CheckboxColumn("Needs review")})
                st.download_button("Download results as CSV", out.to_csv(index=False), "citizensense_results.csv", "text/csv", type="primary")

# ================================================================= TAB 3
with tab_perf:
    st.markdown("### What the system uses")
    st.markdown("""
- **Category, sentiment and priority:** TF-IDF features (words and character groups) with classical models from scikit-learn. The category model is a calibrated linear SVM.
- **Key phrases and summary:** rule-based, so they work offline.
- **Transformers:** zero-shot, multilingual sentiment, summarization and sentence embeddings are available in `app/hf_models.py` and run in the notebook.
""")
    cmp_path, ch_path = ROOT / "outputs" / "model_comparison.csv", ROOT / "outputs" / "challenge_report.txt"
    if cmp_path.exists():
        st.markdown("### Model comparison")
        cmp_df = pd.read_csv(cmp_path).rename(columns={"task": "Task", "model": "Model", "val_accuracy": "Validation accuracy",
                                                       "val_macro_f1": "Validation F1", "test_accuracy": "Test accuracy", "test_macro_f1": "Test F1"})
        st.dataframe(cmp_df.round(3), hide_index=True)
    st.markdown("### How far to trust these numbers")
    st.warning("The test split comes from the same message templates as the training data, so its scores are optimistic. "
               "On 49 free-form messages written by hand, category accuracy drops to about 59%. Short, unusual or multi-issue messages are the weak spots, "
               "and priority is the hardest task. Always have a person confirm urgent cases.")
    cm = ROOT / "outputs" / "confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), caption="Category confusion matrix on the test split", width=560)
    if ch_path.exists():
        with st.expander("Hand-written challenge set report"):
            st.code(ch_path.read_text(), language=None)
