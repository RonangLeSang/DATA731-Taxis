from cleanData import get_clean_df

if __name__ == "__main__":
    path = "Sources/sample.csv"

    df = get_clean_df(path)
    print(df)

