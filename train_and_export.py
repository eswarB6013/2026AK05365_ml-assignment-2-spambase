"""
Train five classifiers on the UCI Spambase dataset and export everything the
Streamlit app needs (models, scaler, held-out test CSV, metrics table).

Usage (BITS Virtual Lab / any machine with internet):
    python train_and_export.py                 # downloads Spambase from UCI
    python train_and_export.py --data spambase.data   # use a local copy instead
"""
import argparse
import io
import os
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 2026
MODEL_DIR = "model"

UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/94/spambase.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
]

WORD_COLS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order",
    "mail", "receive", "will", "people", "report", "addresses", "free", "business",
    "email", "you", "credit", "your", "font", "000", "money", "hp", "hpl", "george",
    "650", "lab", "labs", "telnet", "857", "data", "415", "85", "technology", "1999",
    "parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu",
    "table", "conference",
]
FEATURES = (
    [f"word_freq_{w}" for w in WORD_COLS]
    + ["char_semicolon", "char_paren", "char_bracket", "char_exclam", "char_dollar", "char_hash"]
    + ["capital_run_avg", "capital_run_longest", "capital_run_total"]
)
TARGET = "target"  # 1 = spam, 0 = not spam


def load_spambase(local_path=None):
    if local_path:
        raw = pd.read_csv(local_path, header=None)
    else:
        raw = None
        for url in UCI_URLS:
            try:
                print(f"Downloading {url}")
                payload = urllib.request.urlopen(url, timeout=60).read()
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                        with zf.open("spambase.data") as fh:
                            raw = pd.read_csv(fh, header=None)
                else:
                    raw = pd.read_csv(io.BytesIO(payload), header=None)
                break
            except Exception as exc:  # try next mirror
                print(f"  failed: {exc}")
        if raw is None:
            raise SystemExit("Could not download Spambase. Re-run with --data <path to spambase.data>.")
    raw.columns = FEATURES + [TARGET]
    return raw


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=2.0, gamma="scale", random_state=RANDOM_SEED),
    }


def score_of(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        raw_score = model.predict_proba(X)[:, 1]
    else:
        raw_score = model.decision_function(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, raw_score),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="optional local spambase.data file")
    args = parser.parse_args()

    df = load_spambase(args.data)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features, target = '{TARGET}'")
    print(df[TARGET].value_counts().rename({0: "not spam", 1: "spam"}).to_string(), "\n")

    df.to_csv("spambase.csv", index=False)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_SEED
    )
    test_df.to_csv("test_data.csv", index=False)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    rows = []
    for name, model in build_models().items():
        model.fit(X_train_s, y_train)
        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, file_name))
        rows.append({"ML Model Name": name, **score_of(model, X_test_s, y_test)})
        print(f"trained + saved: {name}")

    table = pd.DataFrame(rows).round(4)
    table.to_csv("metrics.csv", index=False)

    print("\n=== Evaluation on held-out test split ===")
    print(table.to_string(index=False))

    md = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |", "|---|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
    with open("metrics_table.md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\nSaved: model/*.joblib, test_data.csv, spambase.csv, metrics.csv, metrics_table.md")


if __name__ == "__main__":
    main()
