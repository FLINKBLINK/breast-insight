import streamlit as st
from services.data_service import load_data
from data.validation import quality_report

st.title("🛡️ Data Quality")

df = load_data()
st.metric("Duplicados", int(df.duplicated().sum()))
st.dataframe(quality_report(df), use_container_width=True, hide_index=True)
