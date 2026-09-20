import streamlit as st
from database.connection import test_connection

st.set_page_config(page_title="Breast Insight", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.3);
    border-radius: 14px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 Breast Insight")
st.subheader("Plataforma de análise estatística, Bootstrap e Machine Learning")
st.write("Use o menu lateral para explorar os dados, executar reamostragens e avaliar modelos.")

c1, c2, c3 = st.columns(3)
c1.metric("Banco", "SQLite")
c2.metric("Estatística", "Bootstrap")
c3.metric("ML", "Classificação")

st.divider()
st.info("Configure o arquivo .env e execute primeiro a importação da base.")
if st.button("🔌 Testar conexão"):
    ok, msg = test_connection()
    (st.success if ok else st.error)(msg)
