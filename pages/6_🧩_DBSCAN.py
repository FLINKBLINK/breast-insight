import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from services.data_service import load_data


st.title("🧩 Agrupamento com DBSCAN")
st.markdown(
    """
    Esta página aplica o **DBSCAN** (*Density-Based Spatial Clustering of
    Applications with Noise*) à base Breast Cancer Wisconsin Diagnostic.
    O algoritmo identifica regiões densas, encontra grupos sem exigir que a
    quantidade de grupos seja informada antecipadamente e pode marcar
    observações isoladas como ruído.

    **Importante:** a coluna `diagnosis` não é usada para formar os grupos.
    Ela só pode ser comparada aos agrupamentos depois da execução, para fins
    exploratórios e didáticos.
    """
)

df = load_data()

if df.empty:
    st.error("A tabela breast_cancer está vazia. Importe a base na página Database.")
    st.stop()

# Identificadores e diagnóstico não devem participar do cálculo de distâncias.
excluded = {"id", "diagnosis", "diagnosis_label"}
feature_columns = [
    col for col in df.columns
    if col.lower() not in excluded
    and not col.lower().startswith("unnamed")
]

numeric_df = df[feature_columns].select_dtypes(include=[np.number]).copy()
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
numeric_df = numeric_df.dropna(axis=1, how="all")

if numeric_df.shape[1] < 2:
    st.error("Não há pelo menos duas variáveis numéricas válidas para realizar o agrupamento.")
    st.stop()

# Imputa valores ausentes pela mediana e padroniza as variáveis.
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(numeric_df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

st.subheader("1. Preparação dos dados")
c1, c2, c3 = st.columns(3)
c1.metric("Registros", f"{len(df):,}".replace(",", "."))
c2.metric("Variáveis utilizadas", len(numeric_df.columns))
c3.metric("Valores ausentes tratados", int(numeric_df.isna().sum().sum()))
st.caption(
    "As variáveis numéricas foram imputadas pela mediana quando necessário "
    "e padronizadas com StandardScaler. As colunas de diagnóstico e de identificação "
    "foram excluídas das variáveis de entrada."
)

with st.expander("Ver variáveis utilizadas"):
    st.write(", ".join(numeric_df.columns.tolist()))

st.markdown("---")
st.subheader("2. Método do cotovelo: gráfico de distância aos k vizinhos")
st.markdown(
    """
    Para o DBSCAN, o gráfico de **k-distance** é uma forma prática de auxiliar
    a escolha de `eps`. Para cada observação, calculamos a distância até seu
    k-ésimo vizinho mais próximo e ordenamos essas distâncias. Uma mudança
    mais acentuada na curva pode indicar um valor candidato para `eps`.

    Diferentemente do cotovelo tradicional do K-Means (WSS), este gráfico
    auxilia a escolher o raio de vizinhança do DBSCAN; ele não determina
    diretamente a quantidade de grupos.
    """
)

sample_count = len(X_scaled)
default_min_samples = min(5, sample_count)
if sample_count < 3:
    st.warning("A base precisa ter pelo menos três registros para uma análise de agrupamento útil.")
    st.stop()

min_samples = st.slider(
    "min_samples — número mínimo de observações na vizinhança",
    min_value=2,
    max_value=min(20, sample_count),
    value=max(2, default_min_samples),
    help="Inclui a própria observação na contagem de vizinhos."
)

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

fig_kdist = px.line(
    x=np.arange(1, len(k_distances) + 1),
    y=k_distances,
    labels={"x": "Observações ordenadas", "y": f"Distância ao {min_samples}º vizinho"},
    title="Gráfico k-distance para auxiliar a escolha de eps"
)
fig_kdist.update_traces(line={"width": 2})
fig_kdist.update_layout(height=420)
st.plotly_chart(fig_kdist, use_container_width=True)
st.caption(
    "Procure a região em que a curva começa a crescer mais rapidamente. "
    "A leitura do cotovelo é visual e serve como ponto de partida; teste valores "
    "próximos de eps e avalie os grupos e o ruído gerados."
)

st.markdown("---")
st.subheader("3. Configuração e execução do DBSCAN")

col1, col2 = st.columns(2)
with col1:
    eps = st.number_input(
        "eps — raio da vizinhança",
        min_value=0.1,
        max_value=30.0,
        value=5.0,
        step=0.1,
        help="A distância máxima para que duas observações sejam consideradas vizinhas. "
             "Use o gráfico k-distance como referência."
    )
with col2:
    st.metric("min_samples selecionado", min_samples)

if st.button("▶️ Executar DBSCAN", type="primary", use_container_width=True):
    with st.spinner("Calculando os agrupamentos..."):
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = model.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        results = pd.DataFrame({
            "Componente principal 1": X_pca[:, 0],
            "Componente principal 2": X_pca[:, 1],
            "Grupo": labels.astype(int)
        })
        results["Grupo visual"] = results["Grupo"].map(
            lambda value: "Ruído (−1)" if value == -1 else f"Grupo {value}"
        )

        cluster_ids = set(labels) - {-1}
        n_clusters = len(cluster_ids)
        n_noise = int(np.sum(labels == -1))
        noise_pct = n_noise / len(labels) * 100

        st.session_state["dbscan_results"] = results
        st.session_state["dbscan_labels"] = labels
        st.session_state["dbscan_n_clusters"] = n_clusters
        st.session_state["dbscan_n_noise"] = n_noise
        st.session_state["dbscan_eps"] = float(eps)
        st.session_state["dbscan_min_samples"] = int(min_samples)
        st.session_state["dbscan_variance"] = pca.explained_variance_ratio_
        st.session_state["dbscan_noise_pct"] = noise_pct

if "dbscan_results" in st.session_state:
    results = st.session_state["dbscan_results"]
    labels = st.session_state["dbscan_labels"]
    n_clusters = st.session_state["dbscan_n_clusters"]
    n_noise = st.session_state["dbscan_n_noise"]
    variance = st.session_state["dbscan_variance"]

    st.markdown("---")
    st.subheader("4. Resultados do agrupamento")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grupos encontrados", n_clusters)
    m2.metric("Registros classificados como ruído", f"{n_noise:,}".replace(",", "."))
    m3.metric("Percentual de ruído", f"{st.session_state['dbscan_noise_pct']:.1f}%")
    m4.metric("Registros agrupados", f"{len(labels) - n_noise:,}".replace(",", "."))

    if n_clusters == 0:
        st.warning(
            "Com esses parâmetros, o DBSCAN não encontrou grupos: todas as observações "
            "foram classificadas como ruído. Experimente aumentar eps ou reduzir min_samples."
        )
    elif n_clusters == 1:
        st.info(
            "Foi encontrado apenas um grupo. Experimente valores próximos de eps no "
            "gráfico k-distance para investigar se há uma separação mais informativa."
        )

    fig_clusters = px.scatter(
        results,
        x="Componente principal 1",
        y="Componente principal 2",
        color="Grupo visual",
        hover_data={"Grupo": True},
        title="Grupos DBSCAN visualizados com PCA",
        labels={
            "Componente principal 1": f"Componente 1 ({variance[0] * 100:.1f}% da variância)",
            "Componente principal 2": f"Componente 2 ({variance[1] * 100:.1f}% da variância)"
        }
    )
    fig_clusters.update_traces(marker={"size": 8, "opacity": 0.8})
    fig_clusters.update_layout(height=560)
    st.plotly_chart(fig_clusters, use_container_width=True)

    st.caption(
        "O PCA reduz as variáveis para duas dimensões apenas para facilitar a visualização. "
        "O DBSCAN foi aplicado aos dados padronizados completos, não apenas às duas componentes."
    )

    counts = (
        pd.Series(labels, name="Grupo")
        .value_counts()
        .sort_index()
        .rename_axis("Grupo")
        .reset_index(name="Quantidade de registros")
    )
    counts["Grupo"] = counts["Grupo"].map(
        lambda value: "Ruído (−1)" if value == -1 else f"Grupo {value}"
    )
    st.subheader("Tamanho dos grupos")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    # Diagnóstico é consultado apenas após o agrupamento, nunca como entrada do DBSCAN.
    if "diagnosis" in df.columns:
        st.subheader("5. Comparação exploratória com o diagnóstico original")
        st.markdown(
            "Esta tabela é apenas uma comparação posterior. O diagnóstico não foi utilizado "
            "pelo DBSCAN para calcular as distâncias ou criar os grupos."
        )
        comparison = pd.DataFrame({
            "Grupo DBSCAN": [
                "Ruído (−1)" if label == -1 else f"Grupo {label}"
                for label in labels
            ],
            "Diagnóstico original": df["diagnosis"].astype(str).to_numpy()
        })
        cross_tab = pd.crosstab(
            comparison["Grupo DBSCAN"],
            comparison["Diagnóstico original"],
            margins=True
        )
        st.dataframe(cross_tab, use_container_width=True)

    st.subheader("Interpretação")
    st.markdown(
        f"""
        - **eps:** {st.session_state['dbscan_eps']:.2f}
        - **min_samples:** {st.session_state['dbscan_min_samples']}
        - **Grupos encontrados:** {n_clusters}
        - **Ruído:** {n_noise} registros ({st.session_state['dbscan_noise_pct']:.1f}%).

        O DBSCAN agrupa observações em regiões de maior densidade. O rótulo `-1`
        representa observações que o algoritmo não associou a nenhum grupo. O resultado
        depende dos parâmetros e da escala das variáveis; por isso, é importante comparar
        configurações e não interpretar os grupos como categorias clínicas.
        """
    )

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar resultados dos agrupamentos (CSV)",
        data=csv,
        file_name="breast_insight_dbscan_resultados.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.warning(
    "⚠️ Ferramenta educacional para estudo de agrupamento não supervisionado. "
    "Os resultados não representam diagnóstico nem validação clínica."
)
