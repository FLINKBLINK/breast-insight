from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _make_table(data, widths):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172B4D")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _append_plot(story, figure, title, description, styles, width=500, height=280):
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(description, styles["BodyText"]))
    story.append(Spacer(1, 10))
    if figure is None:
        story.append(Paragraph("Gráfico indisponível nesta execução.", styles["BodyText"]))
        return
    try:
        image_bytes = figure.to_image(format="png", width=1200, height=700, scale=1)
        story.append(Image(BytesIO(image_bytes), width=width, height=height))
    except Exception:
        story.append(
            Paragraph(
                "O gráfico foi exibido na aplicação, mas não pôde ser convertido para imagem "
                "neste ambiente. Verifique a instalação do Kaleido/Chrome.",
                styles["BodyText"],
            )
        )
    story.append(Spacer(1, 12))


def generate_dbscan_pdf(
    *,
    records_count,
    feature_count,
    missing_count,
    feature_names,
    eps,
    min_samples,
    n_clusters,
    n_noise,
    noise_pct,
    variance,
    k_distance_figure,
    cluster_figure,
    cluster_counts,
    diagnosis_comparison=None,
):
    """Create a PDF report with the current DBSCAN run and its plots."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=38,
        leftMargin=38,
        topMargin=36,
        bottomMargin=36,
        title="Breast Insight - Relatório DBSCAN",
        author="Breast Insight",
    )
    styles = getSampleStyleSheet()
    styles["Title"].alignment = TA_CENTER
    story = [
        Paragraph("Breast Insight", styles["Title"]),
        Spacer(1, 5),
        Paragraph("Relatório de agrupamento — DBSCAN", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph(
            "Este relatório documenta uma execução do DBSCAN sobre as variáveis "
            "numéricas da base Breast Cancer Wisconsin Diagnostic.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("1. Preparação dos dados", styles["Heading1"]),
        _make_table(
            [
                ["Item", "Resultado"],
                ["Registros analisados", f"{records_count:,}".replace(",", ".")],
                ["Variáveis numéricas utilizadas", str(feature_count)],
                ["Valores ausentes tratados", f"{missing_count:,}".replace(",", ".")],
                ["eps", f"{eps:.3f}"],
                ["min_samples", str(min_samples)],
                ["Grupos encontrados (sem contar ruído)", str(n_clusters)],
                ["Registros classificados como ruído", f"{n_noise:,}".replace(",", ".")],
                ["Percentual de ruído", f"{noise_pct:.1f}%"],
                ["Registros não classificados como ruído", f"{records_count - n_noise:,}".replace(",", ".")],
            ],
            [245, 235],
        ),
        Spacer(1, 10),
        Paragraph(
            "Os valores ausentes foram preenchidos pela mediana e as variáveis foram "
            "padronizadas com StandardScaler. Identificadores e diagnóstico foram "
            "excluídos das variáveis usadas para calcular as distâncias.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Variáveis utilizadas", styles["Heading2"]),
        Paragraph(", ".join(str(name) for name in feature_names), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("2. Método do cotovelo (k-distance)", styles["Heading1"]),
        Paragraph(
            "O gráfico apresenta, em ordem crescente, a distância de cada observação "
            "ao seu k-ésimo vizinho mais próximo. A região em que a curva passa a crescer "
            "mais rapidamente pode servir como referência visual para escolher eps. "
            "A escolha não é automática e deve ser avaliada junto aos agrupamentos obtidos.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]

    _append_plot(
        story,
        k_distance_figure,
        "Gráfico k-distance",
        "Distâncias ordenadas usadas como apoio à escolha do raio de vizinhança.",
        styles,
    )

    story.extend(
        [
            Paragraph("3. Visualização dos agrupamentos", styles["Heading1"]),
            Paragraph(
                "O DBSCAN foi ajustado sobre todas as variáveis numéricas padronizadas. "
                "O PCA abaixo reduz os dados a três componentes apenas para visualização; "
                "não é o espaço usado pelo DBSCAN para formar os grupos.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
        ]
    )
    _append_plot(
        story,
        cluster_figure,
        "Agrupamentos em 3D (PCA)",
        "Cada cor representa um grupo identificado. O rótulo -1 representa observações "
        "que não foram associadas a nenhum grupo e são tratadas como ruído.",
        styles,
        width=500,
        height=300,
    )

    story.extend(
        [
            Paragraph("4. Quantidade de registros por grupo", styles["Heading1"]),
            _make_table(
                [["Grupo", "Quantidade de registros"]]
                + [
                    [str(row["Grupo"]), f'{int(row["Quantidade de registros"]):,}'.replace(",", ".")]
                    for _, row in cluster_counts.iterrows()
                ],
                [280, 200],
            ),
            Spacer(1, 12),
        ]
    )

    if diagnosis_comparison is not None:
        comparison_rows = [["Grupo DBSCAN", "Diagnóstico " + str(diagnosis_comparison.columns[0])]]
        for index, row in diagnosis_comparison.iterrows():
            comparison_rows.append([str(index), " | ".join(f"{col}: {int(row[col])}" for col in diagnosis_comparison.columns)])
        story.extend(
            [
                Paragraph("5. Comparação exploratória com o diagnóstico original", styles["Heading1"]),
                Paragraph(
                    "Esta comparação é posterior ao agrupamento. O diagnóstico não foi "
                    "utilizado para ajustar o DBSCAN e não deve ser interpretado como validação clínica.",
                    styles["BodyText"],
                ),
                Spacer(1, 8),
                _make_table(comparison_rows, [180, 300]),
                Spacer(1, 12),
            ]
        )

    story.extend(
        [
            Paragraph("Interpretação e limitações", styles["Heading1"]),
            Paragraph(
                f"Com eps = {eps:.3f} e min_samples = {min_samples}, o algoritmo encontrou "
                f"{n_clusters} grupo(s) e classificou {n_noise} registro(s) como ruído "
                f"({noise_pct:.1f}%). A proporção de variância explicada pelas três componentes "
                f"do PCA é {variance[0] * 100:.1f}%, {variance[1] * 100:.1f}% e "
                f"{variance[2] * 100:.1f}%, respectivamente.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            Paragraph(
                "O resultado depende dos parâmetros eps e min_samples, bem como da escala "
                "das variáveis. Os grupos são padrões de densidade encontrados nos dados, "
                "não categorias clínicas nem diagnósticos.",
                styles["BodyText"],
            ),
        ]
    )

    doc.build(story)
    return buffer.getvalue()
