from OLS import get_model_values
from cleanData import get_clean_df
from normalisation import normalise, variable_target
import warnings
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def predict_time(day, hour, PULocationID, DOLocationID):
    return -38.9618 * day + 5.8654 * hour + 5.6000 * PULocationID + 42.0028 * DOLocationID


def custom_error_metric(y_true, y_pred, tolerance_seconds):
    errors = abs(y_true - y_pred)
    correct_predictions = errors <= tolerance_seconds
    accuracy = sum(correct_predictions) / len(correct_predictions)
    return accuracy


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    tolerance_seconds = 60

    path = "Sources/sample.csv"

    df = get_clean_df(path)
    df = df.loc[~df['PULocationID'].isin(range(230))]
    df = df.loc[~df['day'].isin(["Sunday", "Saturday"])]
    print(df.head())

    X = df.drop(['trip_duration'], axis=1)

    y = df['trip_duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print(X_train.shape, X_test.shape)
    print(X_train.dtypes)

    encoder = ce.OrdinalEncoder(cols=['day', 'hour', 'PULocationID', 'DOLocationID'])

    X_train = encoder.fit_transform(X_train)

    X_test = encoder.transform(X_test)

    print(X_train.head())

    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy_with_tolerance = custom_error_metric(y_test, y_pred, tolerance_seconds)

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
