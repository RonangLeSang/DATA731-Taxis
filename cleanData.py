import pandas as pd


def clean_df(original_df):
    return pd.DataFrame(original_df[["day", "hour", "PULocationID", "DOLocationID", "trip_duration"]])


def load_df(path):
    return pd.read_csv(path)


def get_day(df):
    df["day"] = df["pickup_datetime"].dt.dayofweek
    return df


def get_hour(df):
    df["hour"] = df["pickup_datetime"].dt.hour


def add_trip_duration(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    df["trip_duration"] = pd.to_numeric(df["trip_duration"], errors='coerce')
    return df


def get_clean_df(path):
    df = load_df(path)
    add_trip_duration(df)
    get_day(df)
    get_hour(df)
    return clean_df(df)
