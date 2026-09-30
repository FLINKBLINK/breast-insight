import streamlit as st
import plotly.graph_objects as go

from services.data_service import load_data
from machine_learning.model import train_model
from machine_learning.pdf_report import generate_pdf


# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.title("🤖 Machine Learning")

st.markdown(
    """
    Esta página permite treinar modelos de classificação
    para os dados de câncer de mama.
    """
)


# ==========================================================
# CARREGAMENTO DOS DADOS
# ==========================================================

df = load_data()


# ==========================================================
# CONFIGURAÇÕES DO MODELO
# ==========================================================

st.subheader("⚙️ Configurações")


selected_model = st.selectbox(
    "Modelo",
    [
        "Logistic Regression",
        "Support Vector Machine (SVM)"
    ]
)


test_size = st.slider(
    "Proporção de teste",
    min_value=0.15,
    max_value=0.40,
    value=0.20,
    step=0.05
)


# ==========================================================
# CONFIGURAÇÕES SVM
# ==========================================================

svm_c = 1.0
svm_kernel = "rbf"
svm_gamma = "scale"


if selected_model == "Support Vector Machine (SVM)":

    st.subheader("🧠 Configurações do SVM")

    svm_c = st.number_input(
        "C",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.1,
        help="Parâmetro de regularização do SVM."
    )

    svm_kernel = st.selectbox(
        "Kernel",
        [
            "rbf",
            "linear",
            "poly",
            "sigmoid"
        ],
        index=0
    )

    svm_gamma = st.selectbox(
        "Gamma",
        [
            "scale",
            "auto"
        ],
        index=0
    )


# ==========================================================
# TREINAMENTO
# ==========================================================

if st.button(
    "🚀 Treinar modelo",
    use_container_width=True
):

    if selected_model == "Support Vector Machine (SVM)":
        model_type = "svm"
    else:
        model_type = "logistic"

    with st.spinner("Treinando modelo..."):

        model, metrics, artifacts = train_model(
            df=df,
            test_size=test_size,
            model_type=model_type,
            svm_c=svm_c,
            svm_kernel=svm_kernel,
            svm_gamma=svm_gamma
        )

    # ======================================================
    # SALVAR RESULTADOS NA SESSION
    # ======================================================

    st.session_state["ml_model"] = model
    st.session_state["ml_metrics"] = metrics
    st.session_state["ml_artifacts"] = artifacts
    st.session_state["ml_model_name"] = selected_model
    st.session_state["ml_test_size"] = test_size
    st.session_state["ml_svm_c"] = svm_c
    st.session_state["ml_svm_kernel"] = svm_kernel
    st.session_state["ml_svm_gamma"] = svm_gamma


# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

if "ml_metrics" in st.session_state:

    metrics = st.session_state["ml_metrics"]

    model_name = st.session_state["ml_model_name"]

    saved_test_size = st.session_state["ml_test_size"]

    saved_svm_c = st.session_state["ml_svm_c"]

    saved_svm_kernel = st.session_state["ml_svm_kernel"]

    saved_svm_gamma = st.session_state["ml_svm_gamma"]


    # ======================================================
    # MÉTRICAS
    # ======================================================

    st.markdown("---")

    st.subheader("📊 Resultados")


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


    # ======================================================
    # MATRIZ DE CONFUSÃO
    # ======================================================

    st.subheader("Matriz de confusão")


    cm = metrics["confusion_matrix"]


    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                "Benigno",
                "Maligno"
            ],
            y=[
                "Benigno",
                "Maligno"
            ],
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


    # ======================================================
    # RELATÓRIO PDF
    # ======================================================

    st.markdown("---")

    st.subheader("📄 Relatório")


    pdf = generate_pdf(
        metrics=metrics,
        model_name=model_name,
        test_size=saved_test_size,
        svm_c=saved_svm_c,
        svm_kernel=saved_svm_kernel,
        svm_gamma=saved_svm_gamma
    )


    st.download_button(
        label="📄 Gerar relatório PDF",
        data=pdf,
        file_name="breast_insight_ml_relatorio.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ==========================================================
# AVISO
# ==========================================================

st.markdown("---")

st.warning(
    "⚠️ Modelo educacional. Os resultados não representam "
    "validação clínica e não devem ser utilizados para diagnóstico."
)
