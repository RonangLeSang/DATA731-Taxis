import statsmodels.api as sm


def get_model_values(x, y):
    est = sm.OLS(x, y).fit()

    print(est.summary())
