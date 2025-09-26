import numpy as np


def rmse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return None

    errors = y_true - y_pred
    squared_errors = errors ** 2
    mean_squared_error = np.mean(squared_errors)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return round(root_mean_squared_error, 3)
