# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy.optimize import minimize_scalar
from decimal import Decimal as dec

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
    variance = np.var(measured_data)
    # -------------------------------------------------------------------
    def likelihood(std):
        L = std ** (alpha - n) * np.exp(- std / beta - 
                n* variance / (2 * (std **2)))
        return -L
    std = minimize_scalar(likelihood, bounds=(0.001, 0.6), 
                    method='bounded', options={'maxiter':19}).x
    
    return std

def interpolate(x, x_pred, y_reg):
    """
    calculates y values from interpolated function y(x)

    Parameters
    ----------
    x : array
        x values for interpolated function
    x_fit : array
        x predictor for interpolation
    y_fit : array
        y regressand predictors of interpolation

    Returns
    -------
    y : array
        interpolated values, same shape as x

    """
    y = np.interp(x, x_pred, y_reg)
    return y

# def step_round(input_vector, step_size):
#     """
#     round value down to custom step_size

#     Parameters
#     ----------
#     input_vector : array
#         array to be rounded
#     step_size : array
#         stepsize for rounding

#     Returns
#     -------
#     rounded_vector : array
#         rounded values

#     """
#     def floor_step_size(input, step):
#         step_size_dec = dec(str(step))
#         rounded_vector = float(int(dec(str(input)) / 
#                             step_size_dec) * step_size_dec)
#         return rounded_vector

#     f = np.vectorize(floor_step_size, otypes=[float], excluded=['stepsize'])
#     return f(input_vector, step_size)

def find_exponent(number):
    """
    find exponent of number in order of ten

    Parameters
    ----------
    number : float
        number of interest

    Returns
    -------
    exponent : int
        exponent in order of ten

    """
    return dec(str(number)).normalize().as_tuple().exponent

# def obj_interpolate(x_pred, y_pred, z_reg):
#     """
#     creates linear interpolation object from x, y predictors
#     and z regressor

#     Parameters
#     ----------
#     x_pred : array
#         x predictor for interpolated function
#     y_pred : array
#         y predictor for interpolated function
#     z_reg : array
#         regressand for interpolation

#     Returns
#     -------
#     y_func(x,y) : scipy.interpolate._interpolate.interp2d
#         scipy interpolation object

#     """
#     func = interp2d(x_pred, y_pred, z_reg)
#     return func
