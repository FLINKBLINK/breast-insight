import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

def train_model(df, test_size=0.2, random_state=42):
    ignored = ["id", "diagnosis", "diagnosis_label"]
    features = [c for c in df.columns if c not in ignored]

    X = df[features]
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y,
        random_state=random_state
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred)
    }

    return pipeline, metrics, (X_test, y_test, pred, proba)
