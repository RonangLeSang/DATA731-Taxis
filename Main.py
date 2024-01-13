from cleanData import clean_df, load_df

if __name__ == "__main__":
    path = "Sources/sample.csv"
    print(clean_df(load_df(path)))
