import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def train_model(
    df,
    test_size=0.2,
    random_state=42,
    model_type="logistic",
    svm_c=1.0,
    svm_kernel="rbf",
    svm_gamma="scale"
):
    """
    Treina um modelo de Machine Learning para classificação
    de diagnóstico benigno/maligno.

    Modelos disponíveis:
    - Logistic Regression
    - Support Vector Machine (SVM)

    Retorna:
    - pipeline treinado
    - métricas
    - artefatos do teste
    """

    ignored = [
        "id",
        "diagnosis",
        "diagnosis_label"
    ]

    features = [
        column
        for column in df.columns
        if column not in ignored
    ]

    X = df[features]
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # ==========================================================
    # DEFINIÇÃO DO MODELO
    # ==========================================================

    if model_type == "svm":

        model = SVC(
            C=svm_c,
            kernel=svm_kernel,
            gamma=svm_gamma,
            probability=True,
            random_state=random_state
        )

    else:

        model = LogisticRegression(
            max_iter=3000,
            random_state=random_state
        )

    # ==========================================================
    # PIPELINE
    # ==========================================================

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            model
        )
    ])

    # ==========================================================
    # TREINAMENTO
    # ==========================================================

    pipeline.fit(
        X_train,
        y_train
    )

    # ==========================================================
    # PREDIÇÕES
    # ==========================================================

    pred = pipeline.predict(X_test)

    proba = pipeline.predict_proba(X_test)[:, 1]

    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            pred
        ),

        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            proba
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            pred
        )
    }

    # ==========================================================
    # RETORNO
    # ==========================================================

    return (
        pipeline,
        metrics,
        (
            X_test,
            y_test,
            pred,
            proba
        )
    )
