import pandas as pd
from sklearn.preprocessing import StandardScaler


def variable_target(df):
    print(df["trip_duration"])
    return df["trip_duration"]


def variable_predictive(df):
    variables = ["day", "hour", "PULocationID", "DOLocationID"]
    return df[variables]


def normalise(df):
    scale = StandardScaler()
    vp = variable_predictive(df)
    df_scaled = scale.fit_transform(vp[["day", "hour", "PULocationID", "DOLocationID"]].values)
    df_scaled = pd.DataFrame(df_scaled, columns=["day", "hour", "PULocationID", "DOLocationID"])
    print(df_scaled)
    return df_scaled
