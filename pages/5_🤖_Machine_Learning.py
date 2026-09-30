import streamlit as st
import plotly.graph_objects as go

from services.data_service import load_data
from machine_learning.model import train_model


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.title("🤖 Machine Learning")

st.write(
    "Treinamento de modelos para classificação "
    "de tumores benignos e malignos."
)


# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================

df = load_data()


# ==========================================
# CONFIGURAÇÕES
# ==========================================

st.subheader("⚙️ Configurações do modelo")


model_type = st.selectbox(
    "Modelo",
    options=[
        "Logistic Regression",
        "SVM"
    ]
)


test_size = st.slider(
    "Proporção de teste",
    0.15,
    0.40,
    0.20,
    0.05
)


# ==========================================
# CONFIGURAÇÕES DO SVM
# ==========================================

svm_c = 1.0
svm_kernel = "rbf"
svm_gamma = "scale"


if model_type == "SVM":

    st.markdown("### 🔧 Parâmetros do SVM")

    col1, col2, col3 = st.columns(3)

    with col1:

        svm_c = st.number_input(
            "C",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1
        )

    with col2:

        svm_kernel = st.selectbox(
            "Kernel",
            options=[
                "rbf",
                "linear",
                "poly",
                "sigmoid"
            ]
        )

    with col3:

        svm_gamma = st.selectbox(
            "Gamma",
            options=[
                "scale",
                "auto"
            ]
        )


# ==========================================
# BOTÃO DE TREINAMENTO
# ==========================================

if st.button(
    "🚀 Treinar modelo",
    use_container_width=True
):

    if model_type == "SVM":
        selected_model = "svm"
    else:
        selected_model = "logistic"


    with st.spinner(
        f"Treinando {model_type}..."
    ):

        model, metrics, artifacts = train_model(

            df,

            test_size=test_size,

            model_type=selected_model,

            svm_c=svm_c,

            svm_kernel=svm_kernel,

            svm_gamma=svm_gamma
        )


    # ======================================
    # RESULTADOS
    # ======================================

    st.success(
        f"{model_type} treinado com sucesso!"
    )


    st.subheader("📊 Métricas")


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


    # ======================================
    # MATRIZ DE CONFUSÃO
    # ======================================

    st.subheader(
        "🧩 Matriz de Confusão"
    )


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


    # ======================================
    # INFORMAÇÕES DO MODELO
    # ======================================

    st.subheader(
        "🔍 Configuração utilizada"
    )


    if selected_model == "svm":

        st.write(
            f"**Modelo:** Support Vector Machine"
        )

        st.write(
            f"**Kernel:** `{svm_kernel}`"
        )

        st.write(
            f"**C:** `{svm_c}`"
        )

        st.write(
            f"**Gamma:** `{svm_gamma}`"
        )

    else:

        st.write(
            "**Modelo:** Logistic Regression"
        )

        st.write(
            "**Max iterations:** `3000`"
        )


# ==========================================
# AVISO
# ==========================================

st.warning(
    "Modelo educacional. "
    "A avaliação não representa validação clínica "
    "nem autoriza uso diagnóstico."
)
