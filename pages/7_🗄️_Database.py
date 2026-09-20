import streamlit as st
from config.settings import DATASET_PATH
from data.ingestion import ingest_dataset
from services.data_service import load_data

st.title("🗄️ Database")

st.write(f"Arquivo esperado: `{DATASET_PATH}`")

if st.button("📥 Importar CSV para SQLite"):
    try:
        df = ingest_dataset(DATASET_PATH)
        load_data.clear()
        st.success(f"Importação concluída: {len(df)} linhas.")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as exc:
        st.error(f"Falha na importação: {exc}")

st.divider()

if st.button("🔎 Consultar tabela"):
    try:
        df = load_data()
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Não foi possível consultar a tabela: {exc}")
