import pandas as pd
from cleanData import get_clean_df
from OLS import get_model_values
from normalisation import normalise, variable_target
from sklearn.metrics import accuracy_score
import warnings
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier


def predict_time(day, hour, PULocationID, DOLocationID):
    return -38.9618 * day + 5.8654 * hour + 5.6000 * PULocationID + 42.0028 * DOLocationID


def custom_error_metric(y_true, y_pred, tolerance_seconds):
    errors = abs(y_true - y_pred)
    correct_predictions = errors <= tolerance_seconds
    accuracy = sum(correct_predictions) / len(correct_predictions)
    return accuracy


def save_results(path):
    chunk_size = 1000
    csv_reader = pd.read_csv(path, chunksize=chunk_size)

    for i, chunk in enumerate(csv_reader):
        mode = 'a' if i > 0 else 'w'
        df = get_clean_df(chunk)
        df.to_csv("Sources/result.csv", mode=mode, index=False, header=(mode == 'w'))
        print(f'Chunk {i + 1} written to {"result.csv"}')

    df = get_clean_df(path)
    df.to_csv("output.csv", index=False)
    return df


def split_x_y(df):
    X = df.drop(['trip_duration'], axis=1)

    y = df['trip_duration']
    return train_test_split(X, y, test_size=0.33, random_state=42)


def encoding(X_train, X_test):
    encoder = ce.OrdinalEncoder(cols=['day', 'hour', 'PULocationID', 'DOLocationID'])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    return X_train, X_test


def predict(X_train, y_train, X_test, y_test, tolerance_seconds):
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return custom_error_metric(y_test, y_pred, tolerance_seconds)


def predict_from_path(path, tolerance_seconds):
    df = pd.read_csv(path)
    X_train, X_test, y_train, y_test = split_x_y(df)
    X_train, X_test = encoding(X_train, X_test)
    return predict(X_train, y_train, X_test, y_test, tolerance_seconds)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    tolerance_seconds = 240

    path = "Sources/result.csv"
    # path = "Sources/sample.csv"
    # path = "Sources/2019_High_Volume_FHV_Trip_Records.csv"

    # df = save_results(path)
    df = pd.read_csv(path)

    print(df.head())

    X_train, X_test, y_train, y_test = split_x_y(df)

    print(X_train.shape, X_test.shape)
    print(X_train.dtypes)

    X_train, X_test = encoding(X_train, X_test)

    print(X_train.head())

    accuracy_with_tolerance = predict(X_train, y_train, X_test, y_test, tolerance_seconds)

    print(f'Model accuracy with tolerance: {accuracy_with_tolerance:.4f}')

    # y = variable_target(df)
    # x = normalise(df)
    # y = y.to_numpy().reshape(-1, 1)
    # y.reshape(200000, 1)
    # get_model_values(x, y)

    # day = 0
    # hour = 11
    # PULocationID = 80
    # DOLocationID = 112
    #
    # print(predict_time(day, hour, PULocationID, DOLocationID))
