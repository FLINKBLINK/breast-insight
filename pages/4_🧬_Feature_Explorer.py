import streamlit as st
import plotly.express as px
from services.data_service import load_data

st.title("🧬 Feature Explorer")

df = load_data()
df["diagnosis_label"] = df["diagnosis"].map({0: "Benigno", 1: "Maligno"})

numeric = [c for c in df.select_dtypes("number").columns if c not in ["id", "diagnosis"]]
x = st.selectbox("Eixo X", numeric, index=0)
y = st.selectbox("Eixo Y", numeric, index=1)

fig = px.scatter(
    df, x=x, y=y, color="diagnosis_label",
    hover_data=["id"], template="plotly_dark",
    title=f"{x} x {y}"
)
st.plotly_chart(fig, use_container_width=True)

corr = df[numeric].corr()
st.subheader("Matriz de correlação")
st.dataframe(corr, use_container_width=True)
