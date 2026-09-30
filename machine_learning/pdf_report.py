from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


def generate_pdf(
    metrics,
    model_name,
    test_size,
    svm_c=None,
    svm_kernel=None,
    svm_gamma=None
):

    buffer = BytesIO()


    # ==========================================================
    # DOCUMENTO
    # ==========================================================

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()


    styles["Title"].alignment = TA_CENTER


    story = []


    # ==========================================================
    # TÍTULO
    # ==========================================================

    story.append(
        Paragraph(
            "Breast Insight",
            styles["Title"]
        )
    )


    story.append(
        Spacer(1, 8)
    )


    story.append(
        Paragraph(
            "Relatório de Machine Learning",
            styles["Heading2"]
        )
    )


    story.append(
        Spacer(1, 20)
    )


    # ==========================================================
    # INFORMAÇÕES DO MODELO
    # ==========================================================

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


    story.append(
        Spacer(1, 20)
    )


    # ==========================================================
    # CONFIGURAÇÕES DO SVM
    # ==========================================================

    if model_name == "Support Vector Machine (SVM)":

        story.append(
            Paragraph(
                "Configurações do SVM",
                styles["Heading2"]
            )
        )


        svm_data = [
            [
                "Parâmetro",
                "Valor"
            ],
            [
                "C",
                str(svm_c)
            ],
            [
                "Kernel",
                str(svm_kernel)
            ],
            [
                "Gamma",
                str(svm_gamma)
            ]
        ]


        table = Table(
            svm_data,
            colWidths=[
                180,
                270
            ]
        )


        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "CENTER"
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )


        story.append(table)


        story.append(
            Spacer(1, 25)
        )


    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    story.append(
        Paragraph(
            "Resultados",
            styles["Heading2"]
        )
    )


    metrics_data = [
        [
            "Métrica",
            "Resultado"
        ],

        [
            "Accuracy",
            f"{metrics['accuracy']:.3%}"
        ],

        [
            "Precision",
            f"{metrics['precision']:.3%}"
        ],

        [
            "Recall",
            f"{metrics['recall']:.3%}"
        ],

        [
            "F1-Score",
            f"{metrics['f1']:.3%}"
        ],

        [
            "ROC-AUC",
            f"{metrics['roc_auc']:.3%}"
        ]
    ]


    table = Table(
        metrics_data,
        colWidths=[
            250,
            200
        ]
    )


    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )


    story.append(table)


    story.append(
        Spacer(1, 25)
    )


    # ==========================================================
    # MATRIZ DE CONFUSÃO
    # ==========================================================

    story.append(
        Paragraph(
            "Matriz de Confusão",
            styles["Heading2"]
        )
    )


    cm = metrics["confusion_matrix"]


    cm_data = [
        [
            "",
            "Predito: Benigno",
            "Predito: Maligno"
        ],

        [
            "Real: Benigno",
            str(cm[0][0]),
            str(cm[0][1])
        ],

        [
            "Real: Maligno",
            str(cm[1][0]),
            str(cm[1][1])
        ]
    ]


    table = Table(
        cm_data,
        colWidths=[
            150,
            150,
            150
        ]
    )


    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                10
            )
        ])
    )


    story.append(table)


    story.append(
        Spacer(1, 25)
    )


    # ==========================================================
    # INTERPRETAÇÃO
    # ==========================================================

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]


    story.append(
        Paragraph(
            "Resumo da matriz de confusão",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            f"O modelo classificou corretamente "
            f"{tn} casos benignos e "
            f"{tp} casos malignos. "
            f"Foram observados "
            f"{fp} falsos positivos e "
            f"{fn} falsos negativos.",
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1, 20)
    )


    # ==========================================================
    # EXPLICAÇÃO DAS MÉTRICAS
    # ==========================================================

    story.append(
        Paragraph(
            "Descrição das métricas",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            "<b>Accuracy:</b> proporção de classificações "
            "corretas em relação ao total de observações.",
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1, 6)
    )


    story.append(
        Paragraph(
            "<b>Precision:</b> proporção das previsões positivas "
            "que realmente pertencem à classe positiva.",
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1, 6)
    )


    story.append(
        Paragraph(
            "<b>Recall:</b> proporção dos casos positivos "
            "identificados corretamente pelo modelo.",
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1, 6)
    )


    story.append(
        Paragraph(
            "<b>F1-Score:</b> média harmônica entre Precision "
            "e Recall.",
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1, 6)
    )


    story.append(
        Paragraph(
            "<b>ROC-AUC:</b> medida da capacidade do modelo "
            "de separar as duas classes.",
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1, 25)
    )


    # ==========================================================
    # AVISO
    # ==========================================================

    story.append(
        Paragraph(
            "<b>Observação:</b> este projeto possui finalidade "
            "educacional. Os resultados não representam "
            "validação clínica e não devem ser utilizados "
            "para diagnóstico.",
            styles["BodyText"]
        )
    )


    # ==========================================================
    # GERAR PDF
    # ==========================================================

    doc.build(story)


    buffer.seek(0)


    return buffer.getvalue()
