from pathlib import Path
import pandas as pd
from sqlalchemy import text
from database.connection import get_engine

def clean_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    df = df.drop_duplicates()
    return df

def ingest_dataset(path):
    df = clean_dataset(path)
    df.to_sql("breast_cancer", get_engine(), if_exists="replace", index=False)
    return df
