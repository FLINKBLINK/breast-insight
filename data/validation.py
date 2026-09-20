import pandas as pd

def quality_report(df):
    rows = []
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "nulls": nulls,
            "null_percent": round(nulls / len(df) * 100, 2),
            "unique_values": int(df[col].nunique())
        })
    return pd.DataFrame(rows)
