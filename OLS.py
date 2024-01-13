import statsmodels.api as sm


def get_model_values(x, y):
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)

    x = sm.add_constant(x)
    est = sm.OLS(y, x).fit()

    print(est.summary())
