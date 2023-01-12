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

def student_t(measured_data, alpha, beta):
    """
    Function for probability updating:
    Computes a posterior predictive sparing factor distribution
    for a list of sparing factors (measured_data) and an
    inverse-gamma conjugate prior distribution given by the
    parameters alpha and beta. Returns a frozen student's t
    random variable posterior

    Parameters
    ----------
    measured_data : list/array
        list of observed sparing factors
    alpha : float
        shape of inverse-gamma distribution
    beta : float
        scale of inverse-gamma distrinbution

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_continuous_frozen
        distribution function

    """
    n_sf = len(measured_data)
    variance = np.var(measured_data)
    mean = np.mean(measured_data)

    alpha_up = alpha + n_sf / 2
    beta_up = beta + variance * n_sf / 2
    student_t = t(df=2 * alpha_up, loc=mean,
        scale=np.sqrt(beta_up / alpha_up))
    return student_t

def fit_gamma_prior(measured_data):
    """
    fits the alpha and beta value for a gamma distribution

    Parameters
    ----------
    measured_data : array
        a nxk matrix with n the amount of patients and k the amount
        of sparing factors per patient

    Returns
    -------
    list
        alpha (shape) and beta (scale) hyperparameter
    """
    stds = np.std(measured_data, axis=1)
    alpha, _, beta = gamma.fit(stds, floc=0)
    return [alpha, beta]

def fit_invgamma_prior(measured_data):
    """
    fits the alpha and beta value for an inverse-gamma distribution

    Parameters
    ----------
    measured_data : array
        a nxk matrix with n the amount of patients and k the amount
        of sparing factors per patient

    Returns
    -------
    list
        alpha (shape) and beta (scale) hyperparameter
    """
    variances = np.var(measured_data, axis=1)
    alpha, _, beta = invgamma.fit(variances, floc=0)
    return [alpha, beta]


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


def std_calc(measured_data, alpha, beta):
    """
    Function for probability updating:
    Computes a maximum a priori estimation of the standard deviation
    for a list of sparing factors (measured_data) and a gamma prior
    distribution given by the parameters alpha and beta

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
    std_updated : float
        most likely std based on the measured data and gamma prior

    """
    n = len(measured_data)
    variance_measured = np.var(measured_data)
    # -------------------------------------------------------------------
    def likelihood(std):
        L = ((std ** (alpha - 1)) / (std ** (n - 1)) * np.exp(- std / beta)
            * np.exp(- n * variance_measured / (2 * (std ** 2)))
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
