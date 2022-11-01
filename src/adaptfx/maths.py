# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import interp1d

def truncated_normal(mean, std, low, upp):
    """
    produces a truncated normal distribution

    Parameters
    ----------
    mean : float, optional
        mean of the sparing factor distribution
    std : float, optional
        standard deviation of the sparing factor distribution
    low : float, optional
        lower bound of distribution
    upp : float, optional
        upper bound of distribution

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        distribution function

    """
    normal = truncnorm((low - mean) / std, (upp - mean) / std,
                        loc=mean, scale=std)

    return normal


def sf_probdist(X, sf_low, sf_high, sf_stepsize, probability_threshold):
    """
    This function produces a probability distribution
    based on the normal distribution X

    Parameters
    ----------
    X : scipy.stats._distn_infrastructure.rv_frozen
        distribution function
    sf_low : float
        lower bound for sparing factor distribution
    sf_high : float
        upper bound for sparing factor distribution
    step_size : float
        sparing factor step size
    probability_threshold : float
        threshold of the probability distribution

    Returns
    -------
    sf : array
        array with sparing factors
    prob : array
        array with probability to each sparing factor

    """
    # create sample sparing factor array
    sample_sf = np.arange(sf_low, sf_high + sf_stepsize, sf_stepsize)
    n_samples = len(sample_sf)

    # sum probability density from cumulative density function in interval
    # to get probability
    half_interval = sf_stepsize/2
    upp_bound = (half_interval - sf_stepsize*1e-6) * np.ones(n_samples)
    low_bound = half_interval * np.ones(n_samples)
    prob = X.cdf(sample_sf + upp_bound) - X.cdf(sample_sf - low_bound)

     # get rid of all probabilities below given threshold
    probability = prob[prob > probability_threshold]
    sf = sample_sf[prob > probability_threshold]
    return [sf, probability]


def std_calc(measured_data, alpha, beta):
    """
    calculates the most likely standard deviation for a list
    of k sparing factors and a gamma conjugate prior

    Parameters
    ----------
    measured_data : list/array
        list/array with k sparing factors
    alpha : float
        shape of gamma distribution
    beta : float
        scale of gamma distrinbution

    Returns
    -------
    std : float
        most likely std based on the measured data and gamma prior

    """
    n = len(measured_data)
    std_values = np.arange(0.00001, 0.5, 0.00001)
    likelihood_values = np.zeros(len(std_values))
    for index, value in enumerate(std_values):
        likelihood_values[index] = (
            value ** (alpha - 1)
            / value ** (n - 1)
            * np.exp(-1 / beta * value)
            * np.exp(-np.var(measured_data) / (2 * (value**2 / n)))
        )  # here i have to check whether it is good.
    std = std_values[np.argmax(likelihood_values)]
    return std

def interpolate(x, x_fit, y_fit, mode='numpy'):
    """
    calculates y values from interpolated function y(x)

    Parameters
    ----------
    x : array
        x values for interpolated function
    x_fit : array
        observables for interpolation
    y_fit : array
        predictors of interpolation

    Returns
    -------
    y : array
        interpolated values, same shape as x

    """
    if mode == 'numpy':
        y = np.interp(x, x_fit, y_fit)
    elif mode == 'scipy':
        y_func = interp1d(x_fit, y_fit)
        y = y_func(x)
    return y