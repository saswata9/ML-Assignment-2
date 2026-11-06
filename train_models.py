"""Train all five classifiers and save model artifacts plus held-out test data."""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
(ROOT / "model").mkdir(exist_ok=True)
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "k_nearest_neighbors": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "gaussian_naive_bayes": GaussianNB(),
    "support_vector_machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, random_state=42))]),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, ROOT / "model" / f"{name}.joblib")
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv(ROOT / "test_data.csv", index=False)
print(f"Saved {len(test_df)} held-out rows and {len(models)} trained models.")
