def correlation_matrix(df):
    return df.select_dtypes("number").corr()
