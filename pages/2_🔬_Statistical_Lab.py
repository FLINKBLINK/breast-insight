import streamlit as st
import plotly.express as px
from services.data_service import load_data
from analytics.descriptive import group_summary

st.title("🔬 Statistical Lab")

df = load_data()
df["diagnosis_label"] = df["diagnosis"].map({0: "Benigno", 1: "Maligno"})

features = [
    c for c in df.select_dtypes("number").columns
    if c not in ["id", "diagnosis"]
]
feature = st.selectbox("Escolha uma variável", features)

summary = group_summary(df, feature)
st.dataframe(summary, use_container_width=True, hide_index=True)

fig = px.box(
    df, x="diagnosis_label", y=feature, color="diagnosis_label",
    points="all", template="plotly_dark",
    title=f"Distribuição de {feature} por grupo"
)
st.plotly_chart(fig, use_container_width=True)
