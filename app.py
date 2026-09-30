"""Streamlit interface for exploring fake job posting detection."""

import pandas as pd
import streamlit as st

from demo_data import build_demo_data
from dataset_loader import load_default_dataset, load_uploaded_dataset
from detector import predict_post, train_and_compare
from storage import get_recent_predictions, initialize_database, save_prediction


initialize_database()


st.set_page_config(
    page_title="Postmark | Job post screening",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
      --paper: #f5f4ee;
      --ink: #172321;
      --muted: #68746f;
      --rule: #d9ded8;
      --green: #195e4d;
      --green-light: #e0eee7;
      --red: #bf4937;
      --red-light: #f8e8e2;
      --gold: #bd7c26;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: var(--paper); color: var(--ink); }
    [data-testid="stAppViewContainer"] { background: repeating-linear-gradient(0deg, transparent 0, transparent 31px, rgba(25,94,77,.025) 32px), var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #e9ede6; border-right: 1px solid var(--rule); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.8rem; }
    .block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: 0 !important; color: var(--ink); }
    h1 { font-size: 2.7rem !important; line-height: 1.1 !important; }
    h2 { font-size: 1.45rem !important; }
    p, label, li { color: var(--ink); }
    .wordmark { font: 700 1.28rem 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--green); }
    .eyebrow { color: var(--green); font: 500 .72rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0; }
    .lede { max-width: 660px; color: var(--muted); font-size: 1.02rem; line-height: 1.65; }
    .rule { border-top: 1px solid var(--rule); margin: 1.1rem 0 1.4rem; }
    .note { background: #e9ede6; border-left: 3px solid var(--green); padding: .75rem .95rem; color: #3e514b; font-size: .84rem; line-height: 1.5; }
    .result { padding: 1.2rem 1.4rem; border: 1px solid var(--rule); background: rgba(255,255,255,.56); border-radius: 5px; }
    .result.fake { border-left: 5px solid var(--red); }
    .result.real { border-left: 5px solid var(--green); }
    .result.review { border-left: 5px solid var(--gold); }
    .result-label { font: 700 1.42rem 'Space Grotesk', sans-serif; margin: .25rem 0; }
    .result-meta { color: var(--muted); font-size: .89rem; }
    .signal { display: inline-block; background: var(--red-light); color: #813528; border-radius: 3px; padding: .28rem .52rem; margin: .2rem .25rem .2rem 0; font-size: .8rem; }
    .signal-clear { background: var(--green-light); color: #235d4d; }
    .small-mono { color: var(--muted); font: .75rem 'DM Mono', monospace; }
    div.stButton > button, div.stFormSubmitButton > button { border-radius: 3px; border: 1px solid var(--green); color: white; background: var(--green); font-weight: 600; min-height: 2.6rem; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { border-color: #114536; background: #114536; color: white; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.55); border: 1px solid var(--rule); border-radius: 4px; padding: .85rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
    .stTabs [aria-selected="true"] { color: var(--green); }
    @media (max-width: 700px) {
      .block-container { padding: 1.4rem 1rem 3rem; }
      h1 { font-size: 2.15rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_example(example_name):
    examples = {
        "Established software role": {
            "title": "Backend software engineer",
            "company_profile": "A public software company with offices in Boston.",
            "location": "Boston, MA",
            "description": "Join our product team to build APIs used by healthcare customers. Includes code review and a technical interview.",
            "requirements": "Three years of Python experience and a degree or equivalent practical experience.",
            "benefits": "Medical, dental, retirement match, and paid time off.",
            "salary_range": "$115,000-$145,000",
        },
        "Suspicious remote offer": {
            "title": "Remote payment processing agent",
            "company_profile": "",
            "location": "Remote",
            "description": "Earn $8,000 weekly with only 30 minutes of work per day. No interview needed. Send your bank login to verify your account today.",
            "requirements": "No experience needed. Immediate start. Provide bank details before onboarding.",
            "benefits": "Guaranteed daily payments.",
            "salary_range": "$8,000 per week",
        },
    }
    for key, value in examples[example_name].items():
        st.session_state[key] = value
    st.session_state.pop("prediction", None)


MODEL_SOURCE_VERSION = 2
if st.session_state.get("model_source_version") != MODEL_SOURCE_VERSION:
    default_dataset = load_default_dataset()
    if default_dataset is None:
        training_frame = build_demo_data()
        training_source = "Synthetic walkthrough examples"
    else:
        training_frame, training_source = default_dataset
    st.session_state.training_results = train_and_compare(training_frame)
    st.session_state.data_source = training_source
    st.session_state.trained_rows = st.session_state.training_results.row_count
    st.session_state.model_source_version = MODEL_SOURCE_VERSION

for field in ("title", "company_profile", "location", "description", "requirements", "benefits", "salary_range"):
    st.session_state.setdefault(field, "")


with st.sidebar:
    st.markdown('<div class="wordmark">POSTMARK <span class="small-mono">/ SCREENING DESK</span></div>', unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("**Train on labeled data**")
    st.caption("Upload a CSV or the Kaggle ZIP. The CSV must include `fraudulent` labels (0 = genuine, 1 = fake).")
    uploaded_file = st.file_uploader("Labeled job-post dataset", type=["csv", "zip"], label_visibility="collapsed")
    if st.button("Train from uploaded dataset", use_container_width=True, disabled=uploaded_file is None):
        try:
            uploaded_frame = load_uploaded_dataset(uploaded_file.name, uploaded_file.getvalue())
            with st.spinner("Comparing three classifiers on a stratified hold-out split..."):
                result = train_and_compare(uploaded_frame)
            st.session_state.training_results = result
            st.session_state.data_source = uploaded_file.name
            st.session_state.trained_rows = result.row_count
            st.session_state.pop("prediction", None)
            st.success(f"Trained on {result.row_count:,} labeled rows.")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("**Current model**")
    results = st.session_state.training_results
    model_name = st.selectbox(
        "Choose a classifier",
        list(results.models),
        index=list(results.models).index(results.recommended_model),
        label_visibility="collapsed",
    )
    st.caption("Default favors recall to surface more suspicious posts for human review.")
    st.caption(f"Training set: {st.session_state.data_source}")
    if st.button("Reload default dataset", use_container_width=True):
        default_dataset = load_default_dataset()
        if default_dataset is None:
            training_frame = build_demo_data()
            training_source = "Synthetic walkthrough examples"
        else:
            training_frame, training_source = default_dataset
        with st.spinner("Training and evaluating on the default dataset..."):
            refreshed_results = train_and_compare(training_frame)
        st.session_state.training_results = refreshed_results
        st.session_state.data_source = training_source
        st.session_state.trained_rows = refreshed_results.row_count
        st.session_state.pop("prediction", None)
        st.rerun()


st.markdown("<div class='eyebrow'>Trust &amp; safety / 01</div>", unsafe_allow_html=True)
st.title("A second look at every listing.")
st.markdown(
    "<p class='lede'>Screen a job post for language associated with employment scams. Review the model estimate alongside visible textual cues, then make the call with human judgment.</p>",
    unsafe_allow_html=True,
)
st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

results = st.session_state.training_results
selected_metrics = results.metrics[results.metrics["Model"] == model_name].iloc[0]
metric_columns = st.columns(3)
metric_columns[0].metric("Labeled posts", f"{st.session_state.trained_rows:,}")
metric_columns[1].metric("Held-out test posts", f"{results.test_row_count:,}")
metric_columns[2].metric("Hold-out F1", f"{selected_metrics['F1']:.1%}")
if "synthetic" in st.session_state.data_source.lower():
    st.warning(
        "These evaluation scores are from the synthetic walkthrough set, not Kaggle. "
        "Upload a labeled dataset or place the Kaggle archive in Downloads, then reload the default dataset."
    )
else:
    st.success(
        f"Evaluating on {st.session_state.data_source}: "
        f"{results.test_row_count:,} held-out posts from {results.row_count:,} labeled rows."
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
screen_tab, lab_tab = st.tabs(["Screen a post", "Model lab"])

with screen_tab:
    st.markdown("### Posting details")
    st.caption("Try a sample post or paste a listing. Sample text is synthetic; the current training source is shown in the sidebar.")
    sample_columns = st.columns([1, 1, 3])
    sample_columns[0].button(
        "Load established role",
        on_click=load_example,
        args=("Established software role",),
    )
    sample_columns[1].button(
        "Load suspicious offer",
        on_click=load_example,
        args=("Suspicious remote offer",),
    )

    with st.form("posting_form"):
        st.text_input("Job title", key="title", placeholder="e.g. Customer support specialist")
        first_row = st.columns(2)
        first_row[0].text_input("Company profile", key="company_profile", placeholder="Who is hiring?")
        first_row[1].text_input("Location", key="location", placeholder="City, region, or remote")
        st.text_area("Job description", key="description", height=155, placeholder="Paste the job description...")
        second_row = st.columns(2)
        second_row[0].text_area("Requirements", key="requirements", height=110)
        second_row[1].text_area("Benefits", key="benefits", height=110)
        st.text_input("Salary range", key="salary_range", placeholder="Optional")
        submitted = st.form_submit_button("Analyze posting", use_container_width=True)

    if submitted:
        record = {key: st.session_state.get(key, "") for key in (
            "title", "company_profile", "location", "description", "requirements", "benefits", "salary_range"
        )}
        try:
            label, score = predict_post(results.models[model_name], record)
            st.session_state.prediction = {"label": label, "score": score, "model": model_name, "record": record}
            save_prediction(record.get("title"), label, score, model_name)
        except ValueError as error:
            st.warning(str(error))

    prediction = st.session_state.get("prediction")
    if prediction:
        score = prediction["score"]
        label = prediction["label"]
        category = "Potentially fraudulent" if label else "No strong scam signal found"
        level = "Elevated risk" if 0.4 <= score <= 0.6 else ("Higher risk" if label else "Lower risk")
        result_class = "review" if 0.4 <= score <= 0.6 else ("fake" if label else "real")
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(
                f"<div class='result {result_class}'><div class='eyebrow'>{level}</div><div class='result-label'>{category}</div><div class='result-meta'>Scored by {prediction['model']}</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Scam-associated language score**")
            st.progress(score)
            st.caption(f"{score:.0%} model score. This is not a calibrated probability that the listing is fraudulent.")
        with right:
            st.markdown("**Textual cues**")
            combined = " ".join(str(value) for value in prediction["record"].values()).lower()
            cues = {
                "Upfront payment": ("pay a fee", "payment required", "purchase our", "registration charge", "course fee"),
                "Sensitive financial details": ("bank details", "bank login", "social security", "account information"),
                "Unusually high pay claims": ("earn $8,000", "$12,000", "$2,000 per day", "$1,500 daily", "guaranteed daily"),
                "Urgency pressure": ("act now", "immediately", "before midnight", "today", "urgent"),
                "Avoids an interview": ("no interview", "without an interview", "no screening"),
            }
            matched = [name for name, phrases in cues.items() if any(phrase in combined for phrase in phrases)]
            if matched:
                st.markdown("".join(f"<span class='signal'>{cue}</span>" for cue in matched), unsafe_allow_html=True)
            else:
                st.markdown("<span class='signal signal-clear'>No listed phrase cues found</span>", unsafe_allow_html=True)
            st.caption("Phrase cues are a separate checklist, not an explanation of the model's decision.")
        st.markdown(
            "<div class='note'><strong>Use this as a screening aid.</strong> Language-only models can miss scams and flag legitimate posts. Verify the employer through its independently found official website; never pay to apply or share passwords or bank logins.</div>",
            unsafe_allow_html=True,
        )

with lab_tab:
    st.markdown("### Classifier comparison")
    st.markdown(
        "Each model is trained on 75% of the labeled rows and evaluated on the stratified 25% hold-out set shown above. The default favors recall to surface more suspicious posts for human review; the ensemble has the highest F1 on this dataset. After comparison, each model is refit on all dataset rows for screening."
    )
    st.caption(
        f"Evaluation source: {st.session_state.data_source} · "
        f"training split: {results.row_count - results.test_row_count:,} · "
        f"test split: {results.test_row_count:,} "
        f"({results.test_genuine_count:,} genuine, {results.test_fraudulent_count:,} fraudulent)."
    )
    display_metrics = results.metrics.copy()
    for column in ("Accuracy", "Precision", "Recall", "F1"):
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.1%}")
    st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    st.caption(f"Current training source: {st.session_state.data_source} · {st.session_state.trained_rows:,} labeled rows")
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("**Expected dataset format**")
    st.markdown(
        "The Kaggle Fake Job Postings dataset uses `fraudulent` as its label (`0` genuine, `1` fraudulent). The app also accepts `is_fraudulent`, `label`, `class`, or `target`, with common text labels such as `fake` / `real`."
    )
    st.download_button(
        "Download synthetic walkthrough CSV",
        data=build_demo_data().to_csv(index=False).encode("utf-8"),
        file_name="synthetic_job_post_examples.csv",
        mime="text/csv",
    )
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("**Recent screenings**")
    recent_predictions = get_recent_predictions()
    if recent_predictions:
        st.dataframe(pd.DataFrame(recent_predictions), hide_index=True, use_container_width=True)
    else:
        st.caption("Completed screenings are recorded here. Original job descriptions are not stored.")
    st.markdown(
        "<div class='note'>The walkthrough CSV is invented for interface testing, not a benchmark or substitute for the Kaggle dataset. Hold-out metrics on a small dataset can vary substantially; check class balance and review false negatives before relying on a model.</div>",
        unsafe_allow_html=True,
    )