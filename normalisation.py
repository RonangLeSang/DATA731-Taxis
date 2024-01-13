from sklearn.preprocessing import StandardScaler
def normalise(df):
    scale = StandardScaler()

    df_scaled = scale.fit_transform(df[["day", "hour", "PULocationID", "DOLocationID"]].as_matrix())
    return df_scaled