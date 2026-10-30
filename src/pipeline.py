from __future__ import annotations

import json
import hashlib
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
REFERENCE_DIR = ROOT / "monitoring"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = RAW_DIR / "Telco-Customer-Churn.csv"
QUALITY_GATE_ROC_AUC = 0.70

TARGET = "Churn"
DROP_COLS = ["customerID", TARGET]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs():
    for folder in [RAW_DIR, MODEL_DIR, REPORT_DIR, REFERENCE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def ingest_data() -> pd.DataFrame:
    ensure_dirs()
    if not DATA_PATH.exists():
        print(f"Downloading dataset from {DATA_URL}")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "ML-Capstone-Portfolio/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response, open(DATA_PATH, "wb") as output:
            shutil.copyfileobj(response, output)
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET, "customerID"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)
    return df


def make_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectPercentile(score_func=f_classif, percentile=80)),
        ("model", model)
    ])


def main():
    ensure_dirs()
    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment("telco-churn-capstone")

    raw = ingest_data()
    df = clean_data(raw)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidates = {
        "logistic_regression": make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced")),
        "random_forest": make_pipeline(RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
        )),
    }
    cv_scores = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_scores[name] = float(scores.mean())
        print(f"{name}: CV ROC-AUC={scores.mean():.4f} +/- {scores.std():.4f}")

    selected_name = max(cv_scores, key=cv_scores.get)
    base = candidates[selected_name]
    best_estimator = base
    best_params = {}

    if selected_name == "random_forest":
        search = RandomizedSearchCV(
            base,
            param_distributions={
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 8, 12, 18],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "select__percentile": [60, 80, 100],
            },
            n_iter=8, scoring="roc_auc", cv=cv, random_state=42, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params = {k: str(v) for k, v in search.best_params_.items()}
        best_cv = float(search.best_score_)
    else:
        search = RandomizedSearchCV(
            base,
            param_distributions={
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                "select__percentile": [60, 80, 100],
            },
            n_iter=6, scoring="roc_auc", cv=cv, random_state=42, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params = {k: str(v) for k, v in search.best_params_.items()}
        best_cv = float(search.best_score_)

    best_estimator.fit(X_train, y_train)
    preds = best_estimator.predict(X_test)
    probabilities = best_estimator.predict_proba(X_test)[:, 1]
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_model": selected_name,
        "cv_roc_auc_by_model": cv_scores,
        "best_cv_roc_auc": best_cv,
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, preds, zero_division=0)),
        "best_params": best_params,
        "quality_gate_roc_auc": QUALITY_GATE_ROC_AUC,
    }
    passed = metrics["test_roc_auc"] >= QUALITY_GATE_ROC_AUC
    metrics["quality_gate_passed"] = bool(passed)
    print(json.dumps(metrics, indent=2))

    with mlflow.start_run(run_name=f"{selected_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("selected_model", selected_name)
        mlflow.log_param("feature_selection", "SelectPercentile(f_classif)")
        mlflow.log_param("quality_gate_roc_auc", QUALITY_GATE_ROC_AUC)
        for key, value in best_params.items():
            mlflow.log_param(key, value)
        for name, value in cv_scores.items():
            mlflow.log_metric(f"cv_roc_auc_{name}", value)
        for key in ["best_cv_roc_auc", "test_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"]:
            mlflow.log_metric(key, metrics[key])
        mlflow.log_metric("dataset_rows", len(df))
        mlflow.log_dict(metrics, "metrics.json")

    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(cv_scores, indent=2), encoding="utf-8")
    df[FEATURES].sample(min(1000, len(df)), random_state=42).to_csv(REFERENCE_DIR / "reference_data.csv", index=False)

    model_path = MODEL_DIR / "model.joblib"
    if passed:
        temp_path = MODEL_DIR / "model_candidate.joblib"
        joblib.dump(best_estimator, temp_path)
        temp_path.replace(model_path)
        metadata = {
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "selected_model": selected_name,
            "test_roc_auc": metrics["test_roc_auc"],
            "features": FEATURES,
        }
        (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Quality gate passed; promoted model to {model_path}")
    elif not model_path.exists():
        print("WARNING: Quality gate failed and no previous model exists. API cannot serve predictions yet.")
    else:
        print("Quality gate failed; keeping the existing model unchanged.")


if __name__ == "__main__":
    main()
