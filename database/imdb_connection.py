from pathlib import Path
import sqlalchemy as sa
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
IMDB_DATABASE_PATH = ROOT / "data" / "imdb_sentiment.db"

@st.cache_resource
def get_imdb_engine():
    IMDB_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sa.create_engine(f"sqlite:///{IMDB_DATABASE_PATH.as_posix()}", future=True,
                            connect_args={"check_same_thread": False})

def imdb_table_exists():
    return sa.inspect(get_imdb_engine()).has_table("imdb_reviews")

def test_imdb_connection():
    try:
        with get_imdb_engine().connect() as connection:
            connection.execute(sa.text("SELECT 1"))
        return True, f"Conexão SQLite independente ativa: {IMDB_DATABASE_PATH.name}"
    except Exception as exc:
        return False, f"Falha na conexão com o banco IMDb: {exc}"
