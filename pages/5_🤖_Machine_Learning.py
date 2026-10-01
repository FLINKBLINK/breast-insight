import streamlit as st
import plotly.graph_objects as go
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from services.data_service import load_data
from machine_learning.model import train_model
from machine_learning.pdf_report import generate_pdf


# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.title("🤖 Machine Learning")

st.markdown(
    """
    Esta página permite treinar modelos de classificação para
    identificar diagnósticos benignos e malignos.
    """
)


# ==========================================================
# CARREGAMENTO DOS DADOS
# ==========================================================

df = load_data()


# ==========================================================
# CONFIGURAÇÕES
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
# CONFIGURAÇÕES DO SVM
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
        help="Controla a penalização dos erros de classificação."
    )

    svm_kernel = st.selectbox(
        "Kernel",
        [
            "rbf",
            "linear",
            "poly",
            "sigmoid"
        ],
        index=0,
        help="Define o tipo de fronteira utilizada pelo SVM."
    )

    svm_gamma = st.selectbox(
        "Gamma",
        [
            "scale",
            "auto"
        ],
        index=0,
        help="Controla a influência de cada observação na fronteira."
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
    # SALVAR RESULTADOS
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
# RESULTADOS
# ==========================================================

if "ml_metrics" in st.session_state:

    metrics = st.session_state["ml_metrics"]

    model_name = st.session_state["ml_model_name"]

    saved_test_size = st.session_state["ml_test_size"]

    saved_svm_c = st.session_state["ml_svm_c"]

    saved_svm_kernel = st.session_state["ml_svm_kernel"]

    saved_svm_gamma = st.session_state["ml_svm_gamma"]

    artifacts = st.session_state["ml_artifacts"]


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

    st.subheader("🔢 Matriz de Confusão")


    cm = metrics["confusion_matrix"]


    fig_cm = go.Figure(
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


    fig_cm.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Classe Predita",
        yaxis_title="Classe Real"
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    # ======================================================
    # GRÁFICO 1
    # SVM COM DUAS CARACTERÍSTICAS
    # ======================================================

    if model_name == "Support Vector Machine (SVM)":

        st.markdown("---")

        st.subheader(
            "🎯 Fronteira de decisão do SVM"
        )

        st.markdown(
            """
            Este gráfico mostra visualmente como o SVM separa
            as duas classes utilizando duas características
            do conjunto de dados.
            """
        )


        feature_x = "radius_mean"
        feature_y = "texture_mean"


        if (
            feature_x in df.columns
            and feature_y in df.columns
        ):

            # --------------------------------------------------
            # DADOS
            # --------------------------------------------------

            X_plot = df[
                [
                    feature_x,
                    feature_y
                ]
            ].copy()


            y_plot = df["diagnosis"]


            # --------------------------------------------------
            # TREINAR SVM APENAS PARA VISUALIZAÇÃO
            # --------------------------------------------------

            scaler_plot = StandardScaler()


            X_scaled = scaler_plot.fit_transform(
                X_plot
            )


            svm_plot = SVC(
                C=saved_svm_c,
                kernel=saved_svm_kernel,
                gamma=saved_svm_gamma
            )


            svm_plot.fit(
                X_scaled,
                y_plot
            )


            # --------------------------------------------------
            # GRID PARA DESENHAR AS REGIÕES
            # --------------------------------------------------

            x_min = X_scaled[:, 0].min() - 1
            x_max = X_scaled[:, 0].max() + 1

            y_min = X_scaled[:, 1].min() - 1
            y_max = X_scaled[:, 1].max() + 1


            xx, yy = np.meshgrid(
                np.linspace(
                    x_min,
                    x_max,
                    250
                ),
                np.linspace(
                    y_min,
                    y_max,
                    250
                )
            )


            grid = np.c_[
                xx.ravel(),
                yy.ravel()
            ]


            Z = svm_plot.predict(
                grid
            )


            Z = Z.reshape(
                xx.shape
            )


            # --------------------------------------------------
            # GRÁFICO
            # --------------------------------------------------

            fig_boundary = go.Figure()


            # Região de decisão
            fig_boundary.add_trace(
                go.Contour(
                    x=np.linspace(
                        x_min,
                        x_max,
                        250
                    ),
                    y=np.linspace(
                        y_min,
                        y_max,
                        250
                    ),
                    z=Z,
                    showscale=False,
                    opacity=0.25,
                    contours=dict(
                        coloring="fill"
                    ),
                    hoverinfo="skip"
                )
            )


            # --------------------------------------------------
            # PONTOS BENIGNOS
            # --------------------------------------------------

            benign = y_plot == 0


            fig_boundary.add_trace(
                go.Scatter(
                    x=X_scaled[benign, 0],
                    y=X_scaled[benign, 1],
                    mode="markers",
                    name="Benigno",
                    marker=dict(
                        size=7,
                        symbol="circle"
                    )
                )
            )


            # --------------------------------------------------
            # PONTOS MALIGNOS
            # --------------------------------------------------

            malignant = y_plot == 1


            fig_boundary.add_trace(
                go.Scatter(
                    x=X_scaled[malignant, 0],
                    y=X_scaled[malignant, 1],
                    mode="markers",
                    name="Maligno",
                    marker=dict(
                        size=7,
                        symbol="x"
                    )
                )
            )


            fig_boundary.update_layout(
                title=(
                    "Divisão das classes pelo SVM — "
                    "Radius × Texture"
                ),
                xaxis_title=feature_x,
                yaxis_title=feature_y,
                height=600
            )


            st.plotly_chart(
                fig_boundary,
                use_container_width=True
            )


            st.caption(
                "Cada ponto representa uma amostra. "
                "As regiões do gráfico representam a classe "
                "predita pelo SVM."
            )


    # ======================================================
    # GRÁFICO 2
    # PCA + SVM
    # ======================================================

    if model_name == "Support Vector Machine (SVM)":

        st.markdown("---")

        st.subheader(
            "🧬 Visualização multivariada — PCA"
        )

        st.markdown(
            """
            O PCA reduz as características do conjunto de dados
            para duas dimensões, permitindo visualizar como os
            grupos benigno e maligno se distribuem.
            """
        )


        # --------------------------------------------------
        # PREPARAR DADOS
        # --------------------------------------------------

        ignored_columns = [
            "id",
            "diagnosis",
            "diagnosis_label"
        ]


        feature_columns = [
            column
            for column in df.columns
            if column not in ignored_columns
        ]


        X_all = df[
            feature_columns
        ].copy()


        y_all = df[
            "diagnosis"
        ]


        # --------------------------------------------------
        # STANDARD SCALER
        # --------------------------------------------------

        scaler_pca = StandardScaler()


        X_scaled_pca = scaler_pca.fit_transform(
            X_all
        )


        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        pca = PCA(
            n_components=2
        )


        X_pca = pca.fit_transform(
            X_scaled_pca
        )


        # --------------------------------------------------
        # SVM NO ESPAÇO PCA
        # --------------------------------------------------

        svm_pca = SVC(
            C=saved_svm_c,
            kernel=saved_svm_kernel,
            gamma=saved_svm_gamma
        )


        svm_pca.fit(
            X_pca,
            y_all
        )


        # --------------------------------------------------
        # GRID
        # --------------------------------------------------

        x_min = X_pca[:, 0].min() - 1
        x_max = X_pca[:, 0].max() + 1

        y_min = X_pca[:, 1].min() - 1
        y_max = X_pca[:, 1].max() + 1


        xx, yy = np.meshgrid(
            np.linspace(
                x_min,
                x_max,
                300
            ),
            np.linspace(
                y_min,
                y_max,
                300
            )
        )


        grid_pca = np.c_[
            xx.ravel(),
            yy.ravel()
        ]


        Z_pca = svm_pca.predict(
            grid_pca
        )


        Z_pca = Z_pca.reshape(
            xx.shape
        )


        # --------------------------------------------------
        # GRÁFICO PCA
        # --------------------------------------------------

        fig_pca = go.Figure()


        # Regiões de decisão
        fig_pca.add_trace(
            go.Contour(
                x=np.linspace(
                    x_min,
                    x_max,
                    300
                ),
                y=np.linspace(
                    y_min,
                    y_max,
                    300
                ),
                z=Z_pca,
                showscale=False,
                opacity=0.25,
                contours=dict(
                    coloring="fill"
                ),
                hoverinfo="skip"
            )
        )


        # --------------------------------------------------
        # BENIGNOS
        # --------------------------------------------------

        benign_pca = y_all == 0


        fig_pca.add_trace(
            go.Scatter(
                x=X_pca[benign_pca, 0],
                y=X_pca[benign_pca, 1],
                mode="markers",
                name="Benigno",
                marker=dict(
                    size=7,
                    symbol="circle"
                )
            )
        )


        # --------------------------------------------------
        # MALIGNOS
        # --------------------------------------------------

        malignant_pca = y_all == 1


        fig_pca.add_trace(
            go.Scatter(
                x=X_pca[malignant_pca, 0],
                y=X_pca[malignant_pca, 1],
                mode="markers",
                name="Maligno",
                marker=dict(
                    size=7,
                    symbol="x"
                )
            )
        )


        # --------------------------------------------------
        # VARIÂNCIA EXPLICADA
        # --------------------------------------------------

        variance_1 = pca.explained_variance_ratio_[0] * 100

        variance_2 = pca.explained_variance_ratio_[1] * 100


        fig_pca.update_layout(
            title="Separação das classes utilizando PCA + SVM",
            xaxis_title=(
                f"Componente Principal 1 "
                f"({variance_1:.1f}% da variância)"
            ),
            yaxis_title=(
                f"Componente Principal 2 "
                f"({variance_2:.1f}% da variância)"
            ),
            height=600
        )


        st.plotly_chart(
            fig_pca,
            use_container_width=True
        )


        st.caption(
            "O PCA transforma as características originais "
            "em duas componentes principais para permitir "
            "a visualização dos dados em duas dimensões."
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
