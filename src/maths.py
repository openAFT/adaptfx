# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import gamma, truncnorm


def data_fit(data):
    """
    This function fits the alpha and beta value for the prior

    Parameters
    ----------
    data : array
        a nxk matrix with n the amount of patients and k the amount
        of sparing factors per patient.

    Returns
    -------
    list
        alpha and beta hyperparameter.
    """
    standard_devs = data.std(axis=1)
    alpha, loc, beta = gamma.fit(standard_devs, floc=0)
    return [alpha, beta]


def get_truncated_normal(mean=0, sd=1, low=0.01, upp=10):
    """
    produces a truncated normal distribution

    Parameters
    ----------
    mean : float, optional
        The default is 0.
    sd : float, optional
        The default is 1.
    low : float, optional
        The default is 0.01.
    upp : float, optional
        The default is 10.

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        distribution function.

    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def probdist(X):
    """
    This function produces a probability distribution
    based on the normal distribution X

    Parameters
    ----------
    X : scipy.stats._distn_infrastructure.rv_frozen
        distribution function.

    Returns
    -------
    prob : list
        list with probabilities for each sparing factor.

    """
    prob = np.zeros(170)
    idx = 0
    for i in np.arange(0.01, 1.71, 0.01):
        prob[idx] = X.cdf(i + 0.004999999999999999999) - X.cdf(i - 0.005)
        idx += 1
    return prob


def distribution_update(sparing_factors, alpha, beta):
    """
    Calculates the probability distributions for all
    fractions based on a 6 sparing factor list

    Parameters
    ----------
    sparing_factors : array/list
        list/array with 6 sparing factors
    alpha : float
        shape of inverse-gamma distribution
    beta : float
        scale of inverse-gamme distrinbution

    Returns
    -------
    list
        means and stds of all 5 fractions.

    """
    means = np.zeros(len(sparing_factors))
    stds = np.zeros(len(sparing_factors))
    for i in range(len(sparing_factors)):
        means[i] = np.mean(sparing_factors[: (i + 1)])
        stds[i] = std_calc(sparing_factors[: (i + 1)], alpha, beta)
    means = np.delete(means, 0)
    stds = np.delete(
        stds, 0
    )   
    # we get rid of the first value as it is only the planning 
    # value and not used in a fraction
    return [means, stds]


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
