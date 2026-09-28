from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Churn_Modelling.csv"
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
NUMERIC = ["CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
CATEGORICAL = ["Geography","Gender"]

def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].copy()
    y = df["Exited"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])

    classifier = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
        batch_size=32, max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
    )

    pipeline = Pipeline([("preprocess", preprocess), ("ann", classifier)])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)

    joblib.dump(pipeline, MODEL_DIR / "churn_ann.joblib")
    metrics = {
        "model": "Artificial Neural Network (MLP)",
        "architecture": [16, 8, 1],
        "test_size": 0.20,
        "random_state": 42,
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "precision_churn": float(report["1"]["precision"]),
        "recall_churn": float(report["1"]["recall"]),
        "f1_churn": float(report["1"]["f1-score"]),
        "churn_rate": float(y.mean()),
        "iterations": int(classifier.n_iter_),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
