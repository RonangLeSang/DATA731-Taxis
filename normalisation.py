from sklearn.preprocessing import StandardScaler
def variableCible(df):
    return df["trip_duration"]
def variablePredictive(df):
    variables= ["day", "hour", "PULocationID", "DOLocationID"]
    return df[variables]
def normalise(df):
    scale = StandardScaler()
    vp=variablePredictive(df)
    df_scaled = scale.fit_transform(vp[["day", "hour", "PULocationID", "DOLocationID"]].as_matrix())
    return df_scaled