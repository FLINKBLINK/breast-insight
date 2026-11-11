import pandas as pd
from database.imdb_connection import get_imdb_engine

REQUIRED_COLUMNS = {"review", "sentiment"}

def prepare_imdb_dataset(source):
    """Read and validate the IMDb CSV without removing repeated reviews."""
    df = pd.read_csv(source)
    normalized = {column.strip().lower(): column for column in df.columns}
    missing = REQUIRED_COLUMNS - set(normalized)
    if missing:
        raise ValueError("O CSV precisa conter as colunas review e sentiment. "
                         f"Colunas ausentes: {', '.join(sorted(missing))}")
    df = df[[normalized["review"], normalized["sentiment"]]].copy()
    df.columns = ["review", "sentiment"]
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    invalid = sorted(set(df["sentiment"]) - {"positive", "negative"})
    if invalid:
        raise ValueError("A coluna sentiment aceita somente positive e negative. "
                         f"Valores encontrados: {', '.join(invalid)}")
    if (df["review"].str.len() == 0).any():
        raise ValueError("Existem avaliações vazias no arquivo.")
    # Preserva todas as linhas, inclusive as duplicadas do CSV original.
    df.insert(0, "review_id", range(1, len(df) + 1))
    return df

def import_imdb_dataset(source):
    df = prepare_imdb_dataset(source)
    df.to_sql("imdb_reviews", get_imdb_engine(), if_exists="replace",
              index=False, chunksize=1000, method="multi")
    return df
