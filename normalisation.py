import pandas as pd
from sklearn.preprocessing import StandardScaler


def variable_target(df):
    return df["trip_duration"]


def variable_predictive(df):
    variables = ["day", "hour", "PULocationID", "DOLocationID"]
    return df[variables]


def normalise(df):
    scale = StandardScaler()
    vp = variable_predictive(df)
    columns_to_scale = ["day", "hour", "PULocationID", "DOLocationID"]

    for column in columns_to_scale:
        vp[column] = pd.to_numeric(vp[column], errors='coerce')

    df_scaled = pd.DataFrame(scale.fit_transform(vp[columns_to_scale]), columns=columns_to_scale)
    return df_scaled
