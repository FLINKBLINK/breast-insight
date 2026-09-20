import streamlit as st
import plotly.graph_objects as go

from services.data_service import load_data
from machine_learning.model import train_model


st.title("🤖 Machine Learning")

df = load_data()

test_size = st.slider(
    "Proporção de teste",
    0.15,
    0.40,
    0.20,
    0.05
)


if st.button("🚀 Treinar modelo"):

    with st.spinner("Treinando modelo..."):
        model, metrics, artifacts = train_model(
            df,
            test_size=test_size
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}"
    )

    c2.metric(
        "Precision",
        f"{metrics['precision']:.3f}"
    )

    c3.metric(
        "Recall",
        f"{metrics['recall']:.3f}"
    )

    c4.metric(
        "F1",
        f"{metrics['f1']:.3f}"
    )

    c5.metric(
        "ROC AUC",
        f"{metrics['roc_auc']:.3f}"
    )

    st.subheader("Matriz de confusão")

    cm = metrics["confusion_matrix"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Benigno", "Maligno"],
            y=["Benigno", "Maligno"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True
        )
    )

    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Classe Predita",
        yaxis_title="Classe Real"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.warning(
    "Modelo educacional. A avaliação não representa validação clínica "
    "nem autoriza uso diagnóstico."
)