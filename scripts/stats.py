# -*- coding: utf-8 -*-

import numpy as np

import logging
logging.basicConfig(format='%(levelname)s: %(module)s.%(funcName)s(): %(message)s')

# -----------------------------------------------------------------------------
# mean error
def me(observation, prediction):
    if len(observation) == len(prediction):
        return np.mean(observation - prediction)
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# mean absolute error
def mae(observation, prediction):
    if len(observation) == len(prediction):
        return np.mean(np.absolute(observation - prediction))
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# mean squared error
def mse(observation, prediction):
    if len(observation) == len(prediction):
        return np.mean(np.power((observation - prediction), 2))
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# root mean squared error
def rmse(observation, prediction):
    if len(observation) == len(prediction):
        return np.sqrt(mse(observation, prediction))
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# pearson's correlation coefficient
def pearson_cc(observation, prediction):
    if len(observation) == len(prediction):
        observation_mean = np.mean(observation)
        observation_mean_diff = np.subtract(observation, observation_mean)

        prediction_mean = np.mean(prediction)
        prediction_mean_diff = np.subtract(prediction, prediction_mean)

        denominator = np.sum(np.multiply(observation_mean_diff, prediction_mean_diff))
        num1 = np.sum(np.power(observation_mean_diff, 2))
        num2 = np.sum(np.power(prediction_mean_diff, 2))

        return denominator / (np.sqrt(np.multiply(num1, num2)))
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# Nash-Sutcliffe efficiency
def nse(observation, prediction):
    if len(observation) == len(prediction):
        observation = np.asarray(observation)
        prediction = np.asarray(prediction)

        return 1 - (
            np.sum((observation - prediction) ** 2, axis=0, dtype=np.float64)
            / np.sum((observation - np.mean(prediction)) ** 2, dtype=np.float64)
        )

    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# Percentage Bias
def pbias(_observation, _prediction):
    if len(observation) == len(prediction):
        observation = np.asarray(_observation)
        prediction = np.asarray(_prediction)

        return (100 * np.sum(observation - prediction, axis=0, dtype=np.float64)
                / np.sum(observation))
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# RMSE-observations standard deviation ratio
def rsr(observation, prediction):
    if len(observation) == len(prediction):
        return rmse(observation, prediction) / np.std(observation)
    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
