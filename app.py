import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

st.set_page_config(page_title="WDBC Model Comparison", page_icon="🧪", layout="wide")
st.title("🧪 Wisconsin Diagnostic Breast Cancer — Classifier Comparison")
st.write("Evaluate five saved classification models on a labelled CSV. The app does not retrain models on uploaded data.")
st.info("Target encoding: 0 = malignant, 1 = benign. Class 1 is treated as positive for precision/recall/F1. Educational demonstration only; not a medical diagnostic tool.")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "model")
# GitHub does not store generated binary artifacts in this source-first repo.
# On first launch, train_models.py deterministically creates the five models
# and the held-out test_data.csv using the documented random seed.
if not all(os.path.exists(os.path.join(MODEL_DIR, filename)) for filename in MODEL_FILES.values()):
    import train_models  # noqa: F401

FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]
@st.cache_resource
def load_models():
    return {name: joblib.load(os.path.join(ROOT, "model", filename))
            for name, filename in MODEL_FILES.items()}
try:
    models = load_models()
except Exception as exc:
    st.error(f"Could not load a saved model: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
st.caption("CSV must contain all 30 input features and a numeric 'target' column. The included test_data.csv is a valid example.")
if uploaded is None:
    st.subheader("Preview of the included test dataset")
    st.dataframe(pd.read_csv(os.path.join(ROOT, "test_data.csv")).head(8), use_container_width=True)
    st.stop()
try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()
missing = [c for c in FEATURES + ["target"] if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()
if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
try:
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df["target"], errors="raise").astype(int)
except Exception as exc:
    st.error(f"All features and target must be numeric. Details: {exc}")
    st.stop()
if not set(y.unique()).issubset({0, 1}):
    st.error("Target must contain only 0 and/or 1 (0 = malignant; 1 = benign).")
    st.stop()

all_results, predictions = [], {}
for name, model in models.items():
    pred = model.predict(X)
    predictions[name] = pred
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    auc = roc_auc_score(y, prob) if prob is not None and y.nunique() == 2 else np.nan
    all_results.append({"Model":name, "Accuracy":accuracy_score(y,pred), "AUC":auc,
                        "Precision":precision_score(y,pred,zero_division=0),
                        "Recall":recall_score(y,pred,zero_division=0),
                        "F1":f1_score(y,pred,zero_division=0),
                        "MCC":matthews_corrcoef(y,pred)})
st.subheader("Comparison of all five models")
st.dataframe(pd.DataFrame(all_results).set_index("Model").style.format("{:.4f}"), use_container_width=True)
selected = st.selectbox("Select a model for detailed results", list(MODEL_FILES))
pred = predictions[selected]
st.subheader(f"Detailed evaluation: {selected}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy_score(y,pred):.4f}")
c2.metric("Precision", f"{precision_score(y,pred,zero_division=0):.4f}")
c3.metric("Recall", f"{recall_score(y,pred,zero_division=0):.4f}")
c4.metric("F1", f"{f1_score(y,pred,zero_division=0):.4f}")
prob = models[selected].predict_proba(X)[:,1] if hasattr(models[selected],"predict_proba") else None
if prob is not None and y.nunique() == 2:
    st.metric("ROC-AUC", f"{roc_auc_score(y,prob):.4f}")
st.metric("Matthews Correlation Coefficient", f"{matthews_corrcoef(y,pred):.4f}")
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=[0,1]),
                       display_labels=["Malignant (0)","Benign (1)"]).plot(ax=ax, colorbar=False, values_format="d")
ax.set_title(f"Confusion matrix — {selected}")
st.pyplot(fig)
plt.close(fig)
st.subheader("Classification report")
report = classification_report(y,pred,labels=[0,1],target_names=["Malignant (0)","Benign (1)"],output_dict=True,zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
out = df.copy()
out["predicted_target"] = pred
st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"),
                   file_name="predictions.csv", mime="text/csv")
