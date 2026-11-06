# Machine Learning Assignment 2 — WDBC Classification

## a. Problem statement
Build and compare five supervised classification models to predict whether a breast mass is malignant or benign from digitized features computed from a fine-needle aspirate image. The project also provides a Streamlit application to evaluate the models on labelled test data.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC).
- **Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic. The dataset is also available through `sklearn.datasets.load_breast_cancer`.
- **Size:** 569 instances and 30 numeric features (meets the assignment minimums of 500 instances and 12 features).
- **Target encoding in this project:** 0 = malignant; 1 = benign.
- **Split:** Stratified 80% train / 20% test, `random_state=42`; 114 test observations.
- **Preprocessing:** StandardScaler is fitted inside pipelines for Logistic Regression, KNN and SVM, avoiding test-set leakage. Decision Tree and Gaussian Naive Bayes are used without feature scaling.

## c. GitHub Repository Link
https://github.com/saswata9/ML-Assignment-2

## d. Models used and evaluation metrics
All five models are trained on the same training split and evaluated on the same held-out test split. Class 1 (benign) is the positive class for Precision, Recall and F1. AUC is calculated from the predicted probability of class 1.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine (RBF) | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.9954) and tied highest accuracy, F1 and MCC. Scaling is included in the pipeline. It is the selected winner on this split. |
| Decision Tree | Most interpretable of the five, but it has the lowest accuracy, AUC, F1 and MCC in this experiment. A single tree can be sensitive to the sample split. |
| K-Nearest Neighbors | Performs strongly, but below Logistic Regression and SVM on this test split. Scaling is important because distance calculations are sensitive to feature magnitude. |
| Gaussian Naive Bayes | Fast probabilistic baseline with high AUC, but lower accuracy and MCC than the leading models. Its conditional-independence and distribution assumptions may not fully describe correlated measurements. |
| Support Vector Machine | RBF SVM ties Logistic Regression on accuracy, precision, recall, F1 and MCC, but has a marginally lower AUC (0.9950 versus 0.9954). |

**Overall winner for this dataset:** Logistic Regression, selected by F1 first and AUC as the tie-breaker. SVM is a very close alternative. Results are specific to this held-out split and should not be interpreted as a clinical performance claim.

## Streamlit application
Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
On first launch, the app runs `train_models.py` if model artifacts are absent. This creates all five saved model files and the 114-row held-out `test_data.csv` using the fixed stratified split (`random_state=42`). The app then loads those artifacts and supports labelled CSV upload, comparison of all five models, a model-selection dropdown, six evaluation metrics, confusion matrix, classification report and downloadable predictions.

The uploaded CSV must contain all 30 feature columns and a numeric `target` column encoded as 0 or 1. The included generated `test_data.csv` is a valid example after the first launch.

## Reproducibility
To train the models and generate the held-out test CSV manually:
```bash
python train_models.py
```

## Important notes
- Streamlit public URL will be added only after actual deployment.
- The BITS Virtual Lab screenshot must be captured by the student in the actual lab environment; no substitute or fabricated screenshot is included.
- This is an educational ML exercise, not a medical diagnostic tool.
