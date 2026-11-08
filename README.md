# Spam Filter Lab - ML Assignment 2

**Name:** Eswar B | **BITS ID:** 2026AK05365

## a. Problem Statement
Classify emails as spam or not spam from word-frequency, character-frequency and capital-letter statistics. Five supervised classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbor, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC). A Streamlit app loads the saved models and evaluates them on an uploaded test CSV without retraining.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository - Spambase
- **Instances:** 4,601 emails (2,788 not spam, 1,813 spam, about 39.4% spam)
- **Features:** 57 numeric (48 `word_freq_*`, 6 `char_*`, 3 `capital_run_*`)
- **Target:** `target` (1 = spam, 0 = not spam), binary classification
- **Split:** 80% train (3,680 rows) / 20% held-out test (921 rows), stratified, random seed 2026. The held-out part is saved as `test_data.csv`.
- **Preprocessing:** StandardScaler fitted on the training split only (`model/scaler.joblib`).

## c. GitHub Repository Link
https://github.com/eswarB6013/ml-assignment-2-spambase

## d. Models Used

All five models were trained on the same standardized training split and evaluated on the same 921-row test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9164 | 0.9645 | 0.9086 | 0.8760 | 0.8920 | 0.8242 |
| Decision Tree | 0.9164 | 0.9149 | 0.8994 | 0.8871 | 0.8932 | 0.8246 |
| KNN | 0.8893 | 0.9490 | 0.8896 | 0.8209 | 0.8539 | 0.7665 |
| Naive Bayes | 0.8208 | 0.9402 | 0.7029 | 0.9449 | 0.8061 | 0.6707 |
| Support Vector Machine | 0.9240 | 0.9732 | 0.9246 | 0.8788 | 0.9011 | 0.8402 |

Model settings: Logistic Regression (max_iter=3000), Decision Tree (max_depth=10), KNN (k=7), Gaussian Naive Bayes (defaults), SVM (RBF kernel, C=2, gamma="scale").

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with 91.64% accuracy, AUC 0.9645 and MCC 0.8242. Precision (0.9086) is higher than recall (0.8760), so it misses some spam but rarely flags legitimate mail. |
| Decision Tree | Matches Logistic Regression's accuracy (91.64%) and has the best F1 of the simple models (0.8932). Its AUC (0.9149) is the lowest of all five models, because a single tree ranks probabilities coarsely. |
| KNN | Weaker than the other models on accuracy (88.93%) and MCC (0.7665). Recall (0.8209) is low, so more spam gets through in the 57-dimensional feature space. |
| Naive Bayes | Highest recall (0.9449), so it catches the most spam, but precision falls to 0.7029 and accuracy to 82.08%. The independence assumption does not hold for correlated word frequencies, which causes many false alarms. |
| Support Vector Machine | Best accuracy (92.40%), AUC (0.9732), precision (0.9246), F1 (0.9011) and MCC (0.8402). The RBF kernel captures the non-linear boundary well. |
| **Overall winner for this dataset** | **Support Vector Machine.** It leads on five of the six metrics, including MCC and AUC, so it is the most balanced. Naive Bayes is preferable only when catching every spam matters more than false alarms. |

## e. Repository Layout
```
app.py                  Streamlit app (CSV upload, model dropdown, metrics, confusion matrix, report)
train_and_export.py     Downloads Spambase, trains 5 models, writes model/ and test_data.csv
requirements.txt
test_data.csv           Held-out test split used by the app
spambase.csv            Full dataset with named columns
metrics.csv, metrics_table.md
model/                  scaler.joblib plus 5 saved models (.joblib)
```

## f. Run Locally
```
pip install -r requirements.txt
python train_and_export.py
streamlit run app.py
```

## g. Streamlit App
Live app: <PASTE YOUR STREAMLIT URL>

The app lets you upload a test CSV (57 feature columns plus `target`), pick a model from a dropdown, view Accuracy, AUC, Precision, Recall, F1 and MCC, and see the confusion matrix, classification report and a comparison of all five models.
