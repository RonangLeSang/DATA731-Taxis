import pandas as pd


def clean_df(original_df):
    return pd.DataFrame(original_df[["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]])


def load_df(path):
    return pd.read_csv(path)
