from OLS import get_model_values
from cleanData import get_clean_df
from normalisation import normalise, variable_target

if __name__ == "__main__":
    path = "Sources/sample.csv"

    df = get_clean_df(path)
    y = variable_target(df)
    x = normalise(df)
    y = y.to_numpy().reshape(-1, 1)
    y.reshape(200000, 1)
    get_model_values(x, y)

