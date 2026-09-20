import pandas as pd
import streamlit as st
from sqlalchemy import text
from database.connection import get_engine

@st.cache_data(ttl=300)
def load_data():
    query = text("SELECT * FROM breast_cancer")
    return pd.read_sql(query, get_engine())
