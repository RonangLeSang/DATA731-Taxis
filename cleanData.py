import pandas as pd


def clean_df(original_df):
    return pd.DataFrame(original_df[["day", "hour", "PULocationID", "DOLocationID", "trip_duration"]])


def load_df(path):
    return pd.read_csv(path)


def get_day(df):
    df["day"] = df["pickup_datetime"].dt.day_name()
    return df


def get_hour(df):
    df["hour"] = df["pickup_datetime"].dt.hour


def add_trip_duration(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()


def location_to_string(df):
    df["PULocationID"].to_string()
    df["DOLocationID"].to_string()


# 259 32
def get_clean_df(df):
    # df = load_df(path)
    add_trip_duration(df)
    get_day(df)
    get_hour(df)
    location_to_string(df)
    df = df.loc[df['PULocationID'].isin([259])]
    df = df.loc[df['DOLocationID'].isin([32])]
    # df = df.loc[~df['day'].isin(["Sunday", "Saturday"])]
    return clean_df(df)
