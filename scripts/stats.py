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
    observation = np.asarray(observation)
    prediction = np.asarray(prediction)

    if len(observation) == len(prediction):
        return 1 - (
            np.sum((observation - prediction) ** 2, axis=0, dtype=np.float64)
            / np.sum((observation - np.mean(prediction)) ** 2, dtype=np.float64)
        )

    else:
        logging.warning('observation and prediction records does not have the same length.')
        return np.nan

# -----------------------------------------------------------------------------
# Percentage Bias
def pbias(observation, prediction):
    observation = np.asarray(observation)
    prediction = np.asarray(prediction)

    if len(observation) == len(prediction):
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
# Kling-Gupta Efficiency (KGE) 2009
# Read more about different in the paper: 
# Brief overview: https://www.rdocumentation.org/packages/hydroGOF/versions/0.4-0/topics/KGE
def kge_2009(observation, prediction, scales=(1, 1, 1), get_all=False):
    # mean
    obs_mean = np.mean(observation)
    pre_mean = np.mean(prediction)

    # std dev
    # unbiased estimator
    obs_sigma = np.std(observation, ddof=1)
    pre_sigma = np.std(prediction, ddof=1)


    # pearson's coefficient
    pr = pearson_cc(observation, prediction)

    # beta (β)
    if obs_mean != 0:
        beta = pre_mean / obs_mean
    else:
        beta = np.nan

    # alpha (α)
    if obs_sigma != 0:
        alpha = pre_sigma / obs_sigma
    else:
        alpha = np.nan

    if not np.isnan(beta) and not np.isnan(alpha):
        ed = np.sqrt(np.power(scales[0] * (pr - 1), 2) +
                     np.power(scales[1] * (alpha - 1), 2) +
                     np.power(scales[2] * (beta - 1), 2)
                     )
        kge = 1 - ed
    else:
        if obs_mean == 0:
            logging.warning('Observation mean is zero, as a result beta is infinite, and so kge cannot be computed.')
        if obs_sigma == 0:
            logging.warning('Observation sigma is zero, as a result alpha is infinite, and so kge cannot be computed.')

        kge = np.nan

    if get_all:
        return pr, alpha, beta, kge
    else:
        return kge

# -----------------------------------------------------------------------------
# Kling-Gupta Efficiency (KGE) 2012
# Read more about different in the paper: 
def kge_2012(observation, prediction, scales=(1, 1, 1), get_all=False):
    # mean
    obs_mean = np.mean(observation)
    pre_mean = np.mean(prediction)

    # std dev
    # unbiased estimator
    obs_sigma = np.std(observation, ddof=1)
    pre_sigma = np.std(prediction, ddof=1)

    # pearson's coefficient
    pr = pearson_cc(observation, prediction)

    # beta (β)
    if obs_mean != 0:
        beta = pre_mean / obs_mean
    else:
        beta = np.nan

    # coefficient of variation (sd / mean)
    if pre_mean != 0:
        pre_cv = pre_sigma / pre_mean
    else:
        pre_cv = np.nan

    if obs_mean != 0:
        obs_cv = obs_sigma / obs_mean
    else:
        obs_cv = np.nan

    # variability ratios
    if not np.isnan(obs_cv):
        gamma = pre_cv / obs_cv
    else:
        gamma = np.nan

    if obs_mean != 0 and pre_mean != 0:
        eds = np.sqrt(np.power(scales[0] * (pr - 1), 2) +
                      np.power(scales[1] * (gamma - 1), 2) +
                      np.power(scales[2] * (beta - 1), 2)
                      )
        kge = 1 - eds
    else:
        if obs_mean == 0:
            logging.warning('Observation mean is zero, as a result observation coefficient of variation is infinite, and so kge cannot be computed.')
        if pre_mean == 0:
            logging.warning('Prediction mean is zero, as a result prediction coefficient of variation is infinite, and so kge cannot be computed.')

        kge = np.nan

    if get_all:
        return pr, gamma, beta, kge
    else:
        return kge

# -----------------------------------------------------------------------------
