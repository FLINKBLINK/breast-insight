import streamlit as st
import plotly.graph_objects as go

from services.data_service import load_data
from machine_learning.model import train_model

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


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

def gerar_pdf(metrics, model_name, test_size, svm_c=None,
              svm_kernel=None, svm_gamma=None):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = []

    # ==========================================
    # TÍTULO
    # ==========================================

    story.append(
        Paragraph(
            "Breast Insight — Relatório de Machine Learning",
            title_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            f"<b>Modelo:</b> {model_name}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Proporção de teste:</b> {test_size:.0%}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 15))

    # ==========================================
    # CONFIGURAÇÕES DO SVM
    # ==========================================

    if model_name == "Support Vector Machine":

        story.append(
            Paragraph(
                "<b>Configurações do SVM</b>",
                styles["Heading2"]
            )
        )

        svm_data = [
            ["Parâmetro", "Valor"],
            ["C", str(svm_c)],
            ["Kernel", str(svm_kernel)],
            ["Gamma", str(svm_gamma)]
        ]

        table = Table(
            svm_data,
            colWidths=[150, 300]
        )

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ])
        )

        story.append(table)

        story.append(Spacer(1, 20))

    # ==========================================
    # MÉTRICAS
    # ==========================================

    story.append(
        Paragraph(
            "<b>Resultados</b>",
            styles["Heading2"]
        )
    )

    metrics_data = [
        ["Métrica", "Resultado"],
        ["Accuracy", f"{metrics['accuracy']:.3%}"],
        ["Precision", f"{metrics['precision']:.3%}"],
        ["Recall", f"{metrics['recall']:.3%}"],
        ["F1-Score", f"{metrics['f1']:.3%}"],
        ["ROC-AUC", f"{metrics['roc_auc']:.3%}"],
    ]

    table = Table(
        metrics_data,
        colWidths=[250, 200]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(table)

    story.append(Spacer(1, 25))

    # ==========================================
    # MATRIZ DE CONFUSÃO
    # ==========================================

    story.append(
        Paragraph(
            "<b>Matriz de Confusão</b>",
            styles["Heading2"]
        )
    )

    cm = metrics["confusion_matrix"]

    cm_data = [
        ["", "Predito: Benigno", "Predito: Maligno"],
        ["Real: Benigno", str(cm[0][0]), str(cm[0][1])],
        ["Real: Maligno", str(cm[1][0]), str(cm[1][1])]
    ]

    table = Table(
        cm_data,
        colWidths=[150, 150, 150]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    story.append(table)

    story.append(Spacer(1, 25))

    # ==========================================
    # INTERPRETAÇÃO
    # ==========================================

    tp = cm[1][1]
    fn = cm[1][0]
    tn = cm[0][0]
    fp = cm[0][1]

    story.append(
        Paragraph(
            "<b>Resumo da matriz de confusão</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"O modelo classificou corretamente {tn} casos benignos "
            f"e {tp} casos malignos. Foram observados {fp} falsos positivos "
            f"e {fn} falsos negativos.",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 20))

    # ==========================================
    # AVISO
    # ==========================================

    story.append(
        Paragraph(
            "<b>Observação:</b> este projeto possui finalidade educacional. "
            "Os resultados não representam validação clínica e não devem "
            "ser utilizados para diagnóstico médico.",
            styles["BodyText"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


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


    st.markdown("---")

pdf = gerar_pdf(
    metrics=metrics,
    model_name="Support Vector Machine",
    test_size=test_size,
    svm_c=svm_c,
    svm_kernel=svm_kernel,
    svm_gamma=svm_gamma
)

st.download_button(
    label="📄 Gerar relatório PDF",
    data=pdf,
    file_name="breast_insight_svm_relatorio.pdf",
    mime="application/pdf",
    use_container_width=True
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
