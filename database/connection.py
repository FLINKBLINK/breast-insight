import sqlalchemy as sa
import streamlit as st
from config.settings import DATABASE_URL

@st.cache_resource
def get_engine():
    return sa.create_engine(DATABASE_URL, future=True)

def test_connection():
    try:
        with get_engine().connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return True, "Conexão com o banco realizada com sucesso."
    except Exception as exc:
        return False, f"Falha na conexão: {exc}"
