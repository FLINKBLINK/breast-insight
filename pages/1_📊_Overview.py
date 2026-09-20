import streamlit as st
import plotly.express as px
from services.data_service import load_data
import plotly.graph_objects as go

st.title("📊 Overview")

try:
    df = load_data()
except Exception as exc:
    st.error(f"Banco ainda não foi populado. Execute a importação em Database. Detalhe: {exc}")
    st.stop()

df["diagnosis_label"] = df["diagnosis"].map({0: "Benigno", 1: "Maligno"})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observações", len(df))
c2.metric("Variáveis", len(df.columns))
c3.metric("Benignos", int((df["diagnosis"] == 0).sum()))
c4.metric("Malignos", int((df["diagnosis"] == 1).sum()))

counts = df["diagnosis_label"].value_counts().reset_index()
counts.columns = ["Diagnóstico", "Quantidade"]

fig = px.pie(counts, names="Diagnóstico", values="Quantidade",
             title="Distribuição dos rótulos", hole=.45, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df.head(20), use_container_width=True, hide_index=True)
