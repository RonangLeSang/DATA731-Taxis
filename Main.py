from OLS import get_model_values
from cleanData import get_clean_df
from normalisation import normalise, variable_target

if __name__ == "__main__":
    path = "Sources/sample.csv"

    df = get_clean_df(path)
    y = variable_target(df)
    print(y)
    x = normalise(df)
    get_model_values(x, y)

