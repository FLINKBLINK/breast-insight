import streamlit as st
import numpy as np
import plotly.express as px
from services.data_service import load_data
from analytics.bootstrap import bootstrap_mean_difference
from config.settings import RANDOM_STATE

st.title("📈 Bootstrap Studio")
st.caption("Intervalo de confiança da diferença entre médias usando reamostragem com reposição.")

df = load_data()
df["diagnosis_label"] = df["diagnosis"].map({0: "Benigno", 1: "Maligno"})

features = [c for c in df.select_dtypes("number").columns if c not in ["id", "diagnosis"]]
feature = st.selectbox("Variável analisada", features, index=features.index("radius_mean") if "radius_mean" in features else 0)
iterations = st.slider("Número de reamostragens", 1000, 20000, 5000, step=1000)
confidence = st.select_slider("Nível de confiança", options=[90, 95, 99], value=95)

a = df.loc[df["diagnosis"] == 1, feature].dropna()
b = df.loc[df["diagnosis"] == 0, feature].dropna()

result = bootstrap_mean_difference(a, b, iterations, RANDOM_STATE)
alpha = (100 - confidence) / 2
low, high = np.percentile(result["distribution"], [alpha, 100 - alpha])

c1, c2, c3 = st.columns(3)
c1.metric("Diferença observada", f"{result['observed']:.4f}")
c2.metric(f"Limite inferior ({confidence}%)", f"{low:.4f}")
c3.metric(f"Limite superior ({confidence}%)", f"{high:.4f}")

fig = px.histogram(
    x=result["distribution"], nbins=60,
    title="Distribuição Bootstrap da diferença de médias",
    labels={"x": "Diferença entre médias"},
    template="plotly_dark"
)
fig.add_vline(x=low, line_dash="dash")
fig.add_vline(x=high, line_dash="dash")
fig.add_vline(x=result["observed"], line_dash="dot")
st.plotly_chart(fig, use_container_width=True)

if low > 0 or high < 0:
    st.success("O intervalo calculado não cruza zero.")
else:
    st.info("O intervalo calculado cruza zero.")

st.warning("Este resultado é estatístico e não constitui diagnóstico clínico.")
