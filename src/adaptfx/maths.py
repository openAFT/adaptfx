# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import t, truncnorm, gamma, invgamma
# from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy.optimize import minimize_scalar
from decimal import Decimal as dec

def truncated_normal(mean, std, low, upp):
    """
    Function for probability distribution:
    returns a frozen truncated normal continuous random variable

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

def student_t(sf_observed, shape_inv, scale_inv):
    """
    Function for probability updating:
    Computes a posterior predictive sparing factor distribution
    for a list of sparing factors (sf_observed) and an inverse-gamma
    conjugate prior distribution given by the shape_inv and scale_inv 
    parameters. Returns a frozen student's t random variable posterior

    Parameters
    ----------
    sf_observed : list/array
        list of observed sparing factors
    shape_inv : float
        shape of inverse-gamma distribution
    scale_inv : float
        scale of inverse-gamma distrinbution

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_continuous_frozen
        distribution function

    """
    n_sf = len(sf_observed)
    variance = np.var(sf_observed)
    mean = np.mean(sf_observed)

    # update the hyperparameters
    shape_prime = shape_inv + n_sf / 2
    # n_sf to correct for normalisation
    scale_prime = scale_inv + n_sf * variance / 2
    # constitue random variable
    student_t = t(df=2 * shape_prime, loc=mean,
        scale=np.sqrt(scale_prime / shape_prime))
    return student_t

def fit_gamma_prior(sf_data):
    """
    fits the shape and scale parameters for a gamma distribution

    Parameters
    ----------
    sf_data : array
        a nxk matrix with n the amount of patients and k the amount
        of sparing factors per patient

    Returns
    -------
    list
        shape and scale hyperparameter
    """
    stds = np.std(sf_data, axis=1)
    shape, _, scale = gamma.fit(stds, floc=0)
    return [shape, scale]

def fit_invgamma_prior(sf_data):
    """
    fits the shape_inv and scale_inv parameters for an
    inverse-gamma distribution

    Parameters
    ----------
    sf_data : array
        a nxk matrix with n the amount of patients and k the amount
        of sparing factors per patient

    Returns
    -------
    list
        shape_inv and scale_inv hyperparameter
    """
    # use variance instead of standard deviation
    variances = np.var(sf_data, axis=1)
    shape_inv, _, scale_inv = invgamma.fit(variances, floc=0)
    return [shape_inv, scale_inv]

def sf_probdist(X, sf_low, sf_high, sf_stepsize, probability_threshold):
    """
    This function computes a probability distribution
    based on the frozen random variable X

    Parameters
    ----------
    X : scipy.stats._distn_infrastructure.rv_frozen
        random variable
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
    list
        of sf, and prob
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

def std_posterior(sf_observed, shape, scale):
    """
    Function for probability updating:
    Computes a maximum a priori estimation of the standard deviation
    for a list of sparing factors (sf_observed) and a gamma prior
    distribution given by the shape and scale parameters

    Parameters
    ----------
    sf_observed : list/array
        list/array with k sparing factors
    shape : float
        shape of gamma distribution
    scale : float
        scale of gamma distrinbution

    Returns
    -------
    std_updated : float
        most likely std based on the observed data and gamma prior

    """
    n = len(sf_observed)
    variance_observed = np.var(sf_observed)
    # -------------------------------------------------------------------
    def likelihood(std):
        L = ((std ** (shape - 1)) / (std ** (n - 1)) * np.exp(- std / scale)
            * np.exp(- n * variance_observed / (2 * (std ** 2)))
        )
        return -L
    std_updated = minimize_scalar(likelihood, bounds=(0.001, 0.6), 
                    method='bounded', options={'maxiter':19}).x
    return std_updated

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