from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn | Model Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "https://ml-capstone-churn-prediction.onrender.com"
).rstrip("/")

SUMMARY_URL = f"{API_URL}/monitoring/summary"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        color: #172554;
    }

    .dashboard-subtitle {
        color: #64748b;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }

    .status-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #ecfdf5;
        color: #047857;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '📊 Customer Churn Monitoring Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Live prediction analytics, model performance and data drift monitoring'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Settings")

st.sidebar.write("**API Server**")

st.sidebar.code(API_URL)

if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.info(
    "This dashboard retrieves monitoring information "
    "directly from the deployed FastAPI service."
)


# ============================================================
# FETCH MONITORING DATA
# ============================================================

@st.cache_data(ttl=30)
def fetch_monitoring_data():

    response = requests.get(
        SUMMARY_URL,
        timeout=90
    )

    response.raise_for_status()

    return response.json()


with st.spinner("Connecting to live prediction API..."):

    try:

        data = fetch_monitoring_data()

    except requests.exceptions.Timeout:

        st.error(
            "API request timed out. Render may be waking up. "
            "Please refresh after some time."
        )

        st.stop()

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the prediction API. "
            "Please check whether the Render service is running."
        )

        st.stop()

    except requests.exceptions.RequestException as error:

        st.error(f"API Error: {error}")

        st.stop()


# ============================================================
# API STATUS
# ============================================================

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=30
    )

    health_response.raise_for_status()

    health = health_response.json()

    if health.get("model_loaded"):

        st.success(
            f"🟢 API Online | Model Version: "
            f"{health.get('model_version', 'Unknown')}"
        )

    else:

        st.warning("API is responding, but model is not loaded.")

except requests.exceptions.RequestException:

    st.warning("Could not verify API health status.")


# ============================================================
# TRAINING METRICS
# ============================================================

metrics = data.get("metrics") or {}

st.divider()

st.subheader("🎯 Model Performance")

if metrics:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test ROC-AUC",
        f"{metrics.get('test_roc_auc', 0):.3f}"
    )

    col2.metric(
        "Test F1 Score",
        f"{metrics.get('test_f1', 0):.3f}"
    )

    col3.metric(
        "Training Rows",
        f"{metrics.get('train_rows', 0):,}"
    )

    quality_passed = metrics.get("quality_gate_passed")

    col4.metric(
        "Quality Gate",
        "PASS" if quality_passed is True
        else "FAIL" if quality_passed is False
        else "N/A"
    )

else:

    st.info(
        "Training metrics are not available. "
        "Check reports/metrics.json in the deployed API."
    )


# ============================================================
# LIVE PREDICTION OVERVIEW
# ============================================================

st.divider()

st.subheader("📈 Live Prediction Overview")

total_requests = int(data.get("total_requests", 0))

churn_yes = int(data.get("churn_yes", 0))

churn_no = int(data.get("churn_no", 0))

average_probability = float(
    data.get("average_probability", 0)
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total API Requests",
    f"{total_requests:,}"
)

col2.metric(
    "Predicted Churn",
    f"{churn_yes:,}"
)

col3.metric(
    "Predicted No Churn",
    f"{churn_no:,}"
)

col4.metric(
    "Average Churn Probability",
    f"{average_probability:.1%}"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

st.divider()

st.subheader("📊 Prediction Distribution")

if total_requests > 0:

    distribution = pd.DataFrame({
        "Prediction": ["Churn Yes", "Churn No"],
        "Count": [churn_yes, churn_no]
    })

    left, right = st.columns(2)

    with left:

        st.bar_chart(
            distribution.set_index("Prediction"),
            y="Count"
        )

    with right:

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "No prediction requests logged yet. "
        "Call POST /predict on the API to generate data."
    )


# ============================================================
# PREDICTION PROBABILITY TREND
# ============================================================

st.divider()

st.subheader("📉 Churn Probability Trend")

predictions = data.get("predictions", [])

if predictions:

    prediction_df = pd.DataFrame(predictions)

    prediction_df["timestamp"] = pd.to_datetime(
        prediction_df["timestamp"],
        errors="coerce",
        utc=True
    )

    prediction_df["probability"] = pd.to_numeric(
        prediction_df["probability"],
        errors="coerce"
    )

    prediction_df = prediction_df.dropna(
        subset=["timestamp", "probability"]
    )

    if not prediction_df.empty:

        trend_df = prediction_df.set_index("timestamp")

        st.line_chart(
            trend_df["probability"],
            y_label="Churn Probability"
        )

else:

    st.info("Prediction trend will appear after API requests.")


# ============================================================
# RECENT PREDICTION RECORDS
# ============================================================

st.divider()

st.subheader("🧾 Recent Predictions")

if predictions:

    recent_df = pd.DataFrame(predictions)

    recent_df["timestamp"] = pd.to_datetime(
        recent_df["timestamp"],
        errors="coerce",
        utc=True
    )

    recent_df = recent_df.sort_values(
        "timestamp",
        ascending=False
    )

    if "probability" in recent_df.columns:

        recent_df["probability"] = (
            pd.to_numeric(
                recent_df["probability"],
                errors="coerce"
            ) * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No recent prediction records available.")


# ============================================================
# FEATURE DRIFT MONITORING
# ============================================================

st.divider()

st.subheader("🔍 Feature Drift Monitoring")

drift_results = data.get("drift", [])

if drift_results:

    drift_df = pd.DataFrame(drift_results)

    st.dataframe(
        drift_df,
        use_container_width=True,
        hide_index=True
    )

    potential_drift = sum(
        1 for item in drift_results
        if item.get("signal") == "Potential drift"
    )

    if potential_drift > 0:

        st.warning(
            f"Potential drift detected in "
            f"{potential_drift} feature(s)."
        )

    else:

        st.success("No clear statistical drift signal detected.")

else:

    st.info(
        "Drift analysis requires at least 5 valid prediction "
        "requests and reference_data.csv in the API deployment."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("🏆 Cross-Validation Model Comparison")

cv_results = metrics.get(
    "cv_roc_auc_by_model",
    {}
)

if cv_results:

    cv_df = pd.DataFrame(
        list(cv_results.items()),
        columns=["Model", "Mean CV ROC-AUC"]
    )

    cv_df = cv_df.set_index("Model")

    st.bar_chart(cv_df)

else:

    st.info(
        "Cross-validation comparison data is not available."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Capstone Project"
)

st.caption(
    "Monitoring uses prediction logs and statistical drift tests. "
    "Live accuracy requires actual ground-truth customer outcomes."
)

st.caption(
    "Data refresh interval: 30 seconds. "
    "Use the refresh button to fetch the latest information."
)
