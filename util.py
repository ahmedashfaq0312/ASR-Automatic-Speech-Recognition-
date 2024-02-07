import json
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

class RULConfig():
    def __init__(self, config_dict:dict):
        assert isinstance(config_dict, dict)
        for key, val in config_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [RULConfig(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, RULConfig(val) if isinstance(val, dict) else val)

def get_config(config_file_path):
    """Reads the config file to a dict and creates an object from it.
    """
    with open(config_file_path, "r") as f:
        config_dict = json.load(f)
    config_obj = RULConfig(config_dict)
    return config_obj

def relative_error(y_test, y_predict, threshold):
    """Calculates the relative error of the prediction.
    """
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re


def root_mean_squared_error(y_test, y_predict):
    """Evaluates the prediction result by calculating the MSE and RMSE.
    """
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return rmse
