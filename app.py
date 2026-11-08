import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Spam Filter Lab", page_icon="📧", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "target"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
        if os.path.exists(os.path.join(MODEL_DIR, fname))
    }
    return scaler, models


def scores_for(model, X, y):
    pred = model.predict(X)
    raw = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return pred, raw, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, raw),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


scaler, models = load_artifacts()

st.title("📧 Spam Filter Lab")
st.caption("Five classifiers on the UCI Spambase dataset (57 features, spam = 1). Models are pre-trained and only evaluated here.")

with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Choose a model", list(models.keys()))
    upload = st.file_uploader("Upload test CSV", type="csv")
    st.markdown("The CSV must hold the 57 feature columns plus a `target` column (1 = spam, 0 = not spam).")

if upload is not None:
    data = pd.read_csv(upload)
    st.success(f"Using uploaded file: {upload.name} ({len(data)} rows)")
elif os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
    st.info(f"No upload yet, showing the bundled test_data.csv ({len(data)} rows).")
else:
    st.warning("Upload a test CSV to start.")
    st.stop()

expected = list(scaler.feature_names_in_)
missing = [c for c in expected + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded CSV is missing columns: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    st.stop()

X = scaler.transform(data[expected])
y = data[TARGET].astype(int)

pred, raw, current = scores_for(models[choice], X, y)

st.subheader(f"Results for {choice}")
cols = st.columns(len(METRIC_NAMES))
for col, name in zip(cols, METRIC_NAMES):
    col.metric(name, f"{current[name]:.4f}")

tab_cm, tab_report, tab_all = st.tabs(["Confusion matrix", "Classification report", "All models compared"])

with tab_cm:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Not spam", "Spam"], cmap="Blues", ax=ax, colorbar=False
    )
    st.pyplot(fig)

with tab_report:
    report = classification_report(y, pred, target_names=["Not spam", "Spam"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

with tab_all:
    rows = []
    for name, mdl in models.items():
        _, _, sc = scores_for(mdl, X, y)
        rows.append({"Model": name, **{k: round(v, 4) for k, v in sc.items()}})
    board = pd.DataFrame(rows).set_index("Model")
    st.dataframe(board)
    st.bar_chart(board[["Accuracy", "AUC", "F1"]])
    best = board["MCC"].idxmax()
    st.write(f"Highest MCC on this file: **{best}** ({board.loc[best, 'MCC']:.4f})")
