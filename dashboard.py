from pathlib import Path
import json

import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "monitoring" / "predictions.csv"
REFERENCE_PATH = ROOT / "monitoring" / "reference_data.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="Churn Model Monitor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn — Model Monitoring")
st.caption("Lightweight portfolio monitor: API request volume, prediction distribution, and numeric-feature drift.")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    a, b, c, d = st.columns(4)
    a.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0):.3f}")
    b.metric("Test F1", f"{metrics.get('test_f1', 0):.3f}")
    c.metric("Training rows", f"{metrics.get('train_rows', 0):,}")
    d.metric("Quality gate", "PASS" if metrics.get("quality_gate_passed") else "FAIL")
    st.subheader("Cross-validation model comparison")
    st.bar_chart(pd.Series(metrics.get("cv_roc_auc_by_model", {}), name="Mean CV ROC-AUC"))
else:
    st.warning("No metrics report yet. Run `python src/pipeline.py` first.")

if not LOG_PATH.exists():
    st.info("No prediction requests logged yet. Call the API's `/predict` endpoint, then refresh this dashboard.")
    st.stop()

logs = pd.read_csv(LOG_PATH)
st.subheader("API traffic")
c1, c2, c3 = st.columns(3)
c1.metric("Total requests", len(logs))
c2.metric("Predicted churn", int((logs["churn_prediction"] == "Yes").sum()))
c3.metric("Average churn probability", f"{logs['churn_probability'].mean():.1%}")
st.line_chart(logs.assign(timestamp=pd.to_datetime(logs["timestamp_utc"], errors="coerce")).set_index("timestamp")["churn_probability"])
st.subheader("Prediction distribution")
st.bar_chart(logs["churn_prediction"].value_counts())

st.subheader("Numeric feature drift (two-sample KS test)")
if REFERENCE_PATH.exists():
    reference = pd.read_csv(REFERENCE_PATH)
    numeric = [c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] if c in logs and c in reference]
    rows = []
    for col in numeric:
        baseline = pd.to_numeric(reference[col], errors="coerce").dropna()
        current = pd.to_numeric(logs[col], errors="coerce").dropna()
        if len(current) >= 5 and len(baseline) >= 5:
            result = ks_2samp(baseline, current)
            rows.append({"feature": col, "baseline_n": len(baseline), "request_n": len(current),
                         "KS statistic": round(float(result.statistic), 4),
                         "p-value": round(float(result.pvalue), 4),
                         "signal": "Potential drift" if result.pvalue < 0.05 else "No clear signal"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("At least five requests are needed before the dashboard calculates a drift signal.")
else:
    st.info("Training reference data not found. Run the training pipeline first.")

st.caption("Drift is a statistical warning, not proof of performance loss. Ground-truth labels are required to measure live model accuracy.")
