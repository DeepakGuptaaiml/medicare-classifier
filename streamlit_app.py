"""Streamlit UI for Medicare Classifier — calls FastAPI backend."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Medicare Classifier", page_icon="🏛️", layout="wide")

st.title("Medicare Classifier")
st.caption(
    "Predict Medicare reportable claims (MMSEA Section 111 / V1M extraction) via FastAPI"
)


@st.cache_data(ttl=300)
def fetch_options() -> dict:
    response = requests.get(f"{API_URL}/model/options", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def fetch_model_info() -> dict:
    response = requests.get(f"{API_URL}/model/info", timeout=10)
    response.raise_for_status()
    return response.json()


def check_health() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code} from {API_URL}/health"
        body = response.json()
        if not body.get("model_loaded"):
            return False, f"API up but model not loaded: {body}"
        return True, ""
    except requests.RequestException as exc:
        return False, str(exc)


def _apply_sample_to_session(sample: dict) -> None:
    for key, value in sample.items():
        st.session_state[key] = value


def load_random_claim() -> None:
    try:
        response = requests.get(f"{API_URL}/model/sample", timeout=10)
        response.raise_for_status()
        sample = response.json()
    except requests.RequestException as exc:
        st.session_state["sample_load_error"] = str(exc)
        return
    st.session_state.pop("sample_load_error", None)
    _apply_sample_to_session(sample)
    st.session_state["loaded_sample"] = sample


def init_form_defaults() -> None:
    defaults = {
        "data_set": "WC",
        "pay_cat": "PI",
        "pay_code": 328,
        "pay_type": "SYS",
        "paid_1": 10000.0,
        "paid_3": 20000.0,
        "amount": 5000.0,
        "proc_unit": 29,
        "cont_num": 5013,
        "claim_open": 1,
        "date_v1m_xmit_flag": 0,
        "is_us_claimant": 1,
        "orm_threshold_met": 1,
        "tpoc_threshold_met": 1,
        "is_wc": 1,
        "pay_code_bucket": "ORM",
        "is_excluded_coverage": 0,
        "is_excluded_line": 0,
        "days_open": 0.0,
        "age_at_event": 65.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


with st.sidebar:
    st.header("API Status")
    st.caption(f"**API_URL:** `{API_URL}`")
    healthy, health_detail = check_health()
    if healthy:
        st.success("Connected to API")
        try:
            info = fetch_model_info()
            st.markdown(f"**Model:** {info['model_name']}")
            st.markdown(f"**Target:** `{info['target']}`")
            metrics = info.get("metrics_test", {})
            if metrics:
                st.markdown("**Test metrics**")
                st.json(metrics)
        except requests.RequestException as exc:
            st.warning(f"Could not load model info: {exc}")
    else:
        st.error("API unavailable")
        st.markdown(f"**Error:** {health_detail}")
        st.stop()

try:
    options = fetch_options()
except requests.RequestException as exc:
    st.error(f"Failed to load form options: {exc}")
    st.stop()

init_form_defaults()

with st.expander("Load sample claim from training data"):
    st.button("Fill form from random claim", key="load_sample_btn", on_click=load_random_claim)
    if err := st.session_state.get("sample_load_error"):
        st.error(f"Could not load sample claim: {err}")
    if sample := st.session_state.get("loaded_sample"):
        st.info("Form filled with a random claim from training data.")
        st.json(sample)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Claim & Pay Code")
    data_set = st.selectbox("Data set", options["data_set"], key="data_set")
    pay_cat = st.selectbox("Pay category", options["pay_cat"], key="pay_cat")
    pay_type = st.selectbox("Pay type", options["pay_type"], key="pay_type")
    pay_code_bucket = st.selectbox(
        "Pay code bucket", options["pay_code_bucket"], key="pay_code_bucket"
    )
    pay_code = st.number_input("Pay code", min_value=0, step=1, key="pay_code")
    proc_unit = st.number_input("Proc unit", min_value=0, step=1, key="proc_unit")
    cont_num = st.number_input("Contract number", min_value=0, step=1, key="cont_num")
    claim_open = st.selectbox("Claim open", [1, 0], key="claim_open")

with col2:
    st.subheader("Payments & Flags")
    paid_1 = st.number_input("Paid 1 (indemnity)", min_value=0.0, key="paid_1")
    paid_3 = st.number_input("Paid 3 (medical)", min_value=0.0, key="paid_3")
    amount = st.number_input("Amount", min_value=0.01, key="amount")
    age_at_event = st.number_input("Age at event", min_value=0.0, max_value=120.0, key="age_at_event")
    days_open = st.number_input("Days open", min_value=0.0, key="days_open")
    date_v1m_xmit_flag = st.selectbox("V1M transmitted", [0, 1], key="date_v1m_xmit_flag")
    is_us_claimant = st.selectbox("US claimant", [1, 0], key="is_us_claimant")
    is_wc = st.selectbox("Workers comp (WC)", [1, 0], key="is_wc")

col3, col4 = st.columns(2)
with col3:
    orm_threshold_met = st.selectbox("ORM threshold met (paid_3 > 750)", [0, 1], key="orm_threshold_met")
    tpoc_threshold_met = st.selectbox("TPOC threshold met (paid_1 > 0)", [0, 1], key="tpoc_threshold_met")
with col4:
    is_excluded_coverage = st.selectbox("Excluded coverage", [0, 1], key="is_excluded_coverage")
    is_excluded_line = st.selectbox("Excluded line", [0, 1], key="is_excluded_line")

payload = {
    "data_set": data_set,
    "pay_cat": pay_cat,
    "pay_code": int(pay_code),
    "pay_type": pay_type,
    "paid_1": float(paid_1),
    "paid_3": float(paid_3),
    "amount": float(amount),
    "proc_unit": int(proc_unit),
    "cont_num": int(cont_num),
    "claim_open": int(claim_open),
    "date_v1m_xmit_flag": int(date_v1m_xmit_flag),
    "is_us_claimant": int(is_us_claimant),
    "orm_threshold_met": int(orm_threshold_met),
    "tpoc_threshold_met": int(tpoc_threshold_met),
    "is_wc": int(is_wc),
    "pay_code_bucket": pay_code_bucket,
    "is_excluded_coverage": int(is_excluded_coverage),
    "is_excluded_line": int(is_excluded_line),
    "days_open": float(days_open),
    "age_at_event": float(age_at_event),
}

st.divider()

if st.button("Classify Claim", type="primary", use_container_width=True):
    with st.spinner("Calling classification API..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                st.code(exc.response.text)
            st.stop()

    if result["is_medicare_reportable"] == 1:
        st.success(f"**{result['label']}** — probability {result['probability']:.1%}")
    else:
        st.warning(f"**{result['label']}** — probability {result['probability']:.1%}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", result["label"])
    m2.metric("Probability", f"{result['probability']:.1%}")
    m3.metric("Model", result.get("model_name", "—"))

    st.subheader("Request payload")
    st.json(payload)

st.markdown("---")
st.caption(
    "Architecture: Streamlit UI → FastAPI `/predict` → `medicare_classifier.pkl` | "
    "Swagger docs at `/docs` on the API server"
)
