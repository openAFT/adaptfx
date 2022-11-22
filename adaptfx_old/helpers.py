# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import gamma

def max_action(bed, actionspace, goal, abt=10):
    """
    Computes the maximal dose that can be delivered to the tumor
    in each fraction depending on the actual accumulated dose

    Parameters
    ----------
    bed : float
        accumulated tumor dose so far.
    actionspace : list/array
        array with all discrete dose steps.
    goal : float
        prescribed tumor dose.
    abt : float, optional
        alpha beta ratio of tumor. The default is 10.

    Returns
    -------
    sizer : integer
        gives the size of the resized actionspace to reach
        the prescribed tumor dose.

    """
    bed_actionspace = bed_calc0(actionspace, abt)
    max_action = min(max(bed_actionspace), goal - bed)
    sizer = np.argmin(np.abs(bed_actionspace - max_action))
    if bed_actionspace[sizer] < max_action:
        # make sure that with prescribing max_action
        # reached dose is not below goal
        sizer += 1

    return sizer

def argfind(bedt, value):
    """
    This function is used to find the index of certain values

    Parameters
    ----------
    bedt : list/array
        list of tumor BED in which value is searched.
    value : float
        item inside list.

    Returns
    -------
    index : integer
        index of value inside list.

    """
    index = min(range(len(bedt)), key=lambda i: abs(bedt[i] - value))
    return index

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
        means and stds of all 5 fractions

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

def data_fit(data):
    """
    This function fits the alpha and beta value for the prior

    Parameters
    ----------
    data : array
        a nxk matrix with n the amount of patients and k the amount
        of sparing factors per patient

    Returns
    -------
    list
        alpha and beta hyperparameter
    """
    standard_devs = data.std(axis=1)
    alpha, _, beta = gamma.fit(standard_devs, floc=0)
    return [alpha, beta]