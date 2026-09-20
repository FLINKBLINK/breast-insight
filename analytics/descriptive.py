import pandas as pd

def group_summary(df, feature):
    return (
        df.groupby("diagnosis_label")[feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
