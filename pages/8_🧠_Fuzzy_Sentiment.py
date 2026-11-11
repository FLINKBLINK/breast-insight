import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from data.imdb_ingestion import import_imdb_dataset
from database.imdb_connection import get_imdb_engine, imdb_table_exists, test_imdb_connection
from sentiment.fuzzy_model import analyze_review, analyze_dataset

st.title("🧠 Análise de sentimentos com lógica fuzzy")
st.caption("IMDb Reviews · VADER + inferência fuzzy Mamdani · SQLite independente")
st.markdown(
    "O VADER extrai polaridade e intensidade lexical; um sistema fuzzy Mamdani "
    "combina essas evidências por regras linguísticas e produz um índice contínuo "
    "de favorabilidade entre 0 e 100."
)
st.info("Módulo independente da base Breast Cancer Wisconsin. Os dados ficam na "
        "tabela imdb_reviews, no arquivo data/imdb_sentiment.db.")

if not imdb_table_exists():
    st.subheader("1. Importar a base IMDb")
    uploaded = st.file_uploader("Envie o arquivo IMDB Dataset.csv", type=["csv"])
    if uploaded is not None and st.button("Importar para o SQLite IMDb", type="primary"):
        try:
            with st.spinner("Validando e importando as avaliações..."):
                imported = import_imdb_dataset(uploaded)
            st.success(f"Importação concluída: {len(imported):,} avaliações.".replace(",", "."))
            st.rerun()
        except Exception as exc:
            st.error(f"Não foi possível importar o arquivo: {exc}")
    st.warning("O banco é criado após o envio do CSV. Em hospedagens com sistema de "
               "arquivos efêmero, pode ser necessário importar novamente após uma "
               "reinicialização do aplicativo.")
    st.stop()

try:
    with get_imdb_engine().connect() as connection:
        df = pd.read_sql("SELECT * FROM imdb_reviews", connection)
except Exception as exc:
    st.error(f"Não foi possível consultar o banco IMDb: {exc}")
    st.stop()

st.subheader("1. Visão geral da base")
positive = int((df["sentiment"] == "positive").sum())
negative = int((df["sentiment"] == "negative").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avaliações", f"{len(df):,}".replace(",", "."))
c2.metric("Positivas", f"{positive:,}".replace(",", "."))
c3.metric("Negativas", f"{negative:,}".replace(",", "."))
c4.metric("Duplicatas preservadas", f"{int(df['review'].duplicated().sum()):,}".replace(",", "."))
counts = df["sentiment"].value_counts().rename_axis("Sentimento").reset_index(name="Quantidade")
fig = px.pie(counts, names="Sentimento", values="Quantidade", hole=0.5,
             title="Distribuição dos sentimentos originais", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("2. Como funciona o sistema fuzzy")
with st.expander("Variáveis linguísticas e regras de inferência", expanded=True):
    st.markdown(
        "**Entradas:** polaridade compound do VADER (-1 a 1) e intensidade lexical "
        "(proporção de termos positivos + negativos, limitada a 0–1). "
        "**Conjuntos de entrada:** polaridade negativa, neutra e positiva; "
        "intensidade baixa, média e alta. **Saída:** favorabilidade de 0 a 100, "
        "com inferência Mamdani (min/máx) e defuzzificação pelo centroide. "
        "Exemplos: SE polaridade é negativa E intensidade é média/alta, ENTÃO "
        "favorabilidade é negativa; SE polaridade é positiva E intensidade é "
        "média/alta, ENTÃO favorabilidade é positiva; evidência lexical baixa "
        "aproxima a saída de neutra."
    )
st.caption("Os limites das funções de pertinência são parâmetros didáticos explícitos, "
           "não probabilidades calibradas.")

st.divider()
st.subheader("3. Avaliação fuzzy na base rotulada")
st.write("O índice fuzzy é convertido em classe para comparação: abaixo de 50 = "
         "negativa e a partir de 50 = positiva. A faixa neutra, quando exibida na "
         "análise individual, é tratada como erro nas métricas binárias.")
sample_limit = st.select_slider(
    "Quantidade de avaliações para executar",
    options=[1000, 5000, 10000, 25000, len(df)],
    value=min(5000, len(df)),
    format_func=lambda value: f"{value:,}".replace(",", "."),
)
if st.button("Executar análise fuzzy", type="primary"):
    sample = df.head(int(sample_limit)).copy()
    with st.spinner("Calculando polaridade e inferência fuzzy..."):
        st.session_state["imdb_fuzzy_results"] = analyze_dataset(sample)

if "imdb_fuzzy_results" in st.session_state:
    analyzed = st.session_state["imdb_fuzzy_results"].copy()
    y_true = (analyzed["sentiment"] == "positive").astype(int)
    y_pred = (analyzed["fuzzy_score"] >= 50).astype(int)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    m3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    m4.metric("F1-score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig_cm = go.Figure(data=go.Heatmap(
        z=matrix, x=["Predito negativo", "Predito positivo"],
        y=["Real negativo", "Real positivo"], text=matrix,
        texttemplate="%{text}", colorscale="Blues", showscale=True))
    fig_cm.update_layout(title="Matriz de confusão", height=420)
    st.plotly_chart(fig_cm, use_container_width=True)
    fig_scores = px.histogram(
        analyzed, x="fuzzy_score", color="sentiment", nbins=40,
        barmode="overlay", opacity=0.72,
        title="Distribuição do índice fuzzy por sentimento original",
        labels={"fuzzy_score": "Favorabilidade fuzzy (0–100)"},
        template="plotly_dark")
    st.plotly_chart(fig_scores, use_container_width=True)
    report = classification_report(y_true, y_pred, labels=[0, 1],
        target_names=["negative", "positive"], output_dict=True, zero_division=0)
    st.subheader("Relatório de classificação")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    st.caption("Métricas calculadas sobre a amostra selecionada, na ordem original "
               "do CSV. Para avaliação acadêmica rigorosa, ajuste funções fuzzy "
               "em dados de desenvolvimento e reporte resultados em conjunto "
               "separado, evitando avaliar nos mesmos dados usados no ajuste.")

st.divider()
st.subheader("4. Analisar uma avaliação individual")
review = st.text_area("Cole uma avaliação em inglês", height=180,
                      placeholder="I really enjoyed this movie because...")
if st.button("Analisar texto"):
    if not review.strip():
        st.warning("Digite uma avaliação antes de analisar.")
    else:
        result = analyze_review(review)
        a, b, c = st.columns(3)
        a.metric("Polaridade VADER", f"{result['polarity']:.3f}")
        b.metric("Intensidade lexical", f"{result['intensity']:.3f}")
        c.metric("Favorabilidade fuzzy", f"{result['fuzzy_score']:.1f}/100")
        st.write(f"Classe fuzzy: **{result['fuzzy_class']}**")
        st.progress(int(round(result["fuzzy_score"])))
        st.caption("A saída fuzzy é produzida por regras e não representa probabilidade.")
st.divider()
ok, message = test_imdb_connection()
(st.success if ok else st.error)(message)
st.warning("Limitação: VADER foi desenvolvido para texto curto e informal. "
           "Avaliações longas do IMDb podem conter ironia, contexto e opiniões "
           "mistas que um léxico não compreende plenamente. O sistema é "
           "exploratório, não um modelo de linguagem.")
