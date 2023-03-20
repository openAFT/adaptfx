# -*- coding: utf-8 -*-
import numpy as np
import adaptfx as afx
import scipy.optimize as opt

def bed_calc0(dose, ab, sf=1):
    """
    calculates the BED for a specific dose

    Parameters
    ----------
    dose : float
        physical dose to be delivered.
    ab : float
        alpha-beta ratio of tissue.
    sf : float, optional
        sparing factor. The default is 1 (tumor).

    Returns
    -------
    BED : float
        BED to be delivered based on dose, sparing factor
        and alpha-beta ratio.

    """
    BED = sf * dose * (1 + (sf * dose) / ab)
    return BED


def bed_calc_matrix(actionspace, ab, sf):
    """
    calculates the BED for an array of values

    Parameters
    ----------
    actionspace : list/array
        doses to be delivered.
    ab : float
        alpha-beta ratio of tissue.
    sf : list/array
        list of sparing factors to calculate the correspondent BED.

    Returns
    -------
    BED : List/array
        list of all future BEDs based on the delivered doses
        and sparing factors.

    """
    BED = np.outer(actionspace, sf) * (
        1 + np.outer(actionspace, sf) / ab
    )  # produces a actions space x sparing factor array
    return BED

def convert_to_physical(bed, ab, sf=1):
    """
    Converts given BED to the physical dose

    Parameters
    ----------
    BED : float/array
        tumor or OAR BED for which physical dose has to be calculated.
    ab : float
        alpha-beta ratio.
    sf : float/array
        sparing factor, only specify when OAR BED.

    Returns
    -------
    dose : positive values float/array
        physical dose
    """
    bed_array = np.array(bed)
    bed_array[bed_array < 0] = 0
    physical_dose = (-sf + np.sqrt(sf**2 + 4 * sf**2 * bed_array / ab)) / (
        2 * sf**2 / ab)
    return physical_dose

def cost_func(keys, n_list, n_samples):
    """
    For a specified list of maximum number of fractions
    simulates average cumulative OAR BED for uniform-,
    adaptive- and optimal-fractionation (theoretical optimum)

    Parameters
    ----------
    keys : dict
        algorithm instructions.
    n_list : array
        array of maximum number of fractions
    n_samples : int
        number of patients to sample

    Returns
    -------
    uft : array
        uniform fractionated average cumulative BED
    aft : array
        adaptive fractionated average cumulative BED
    opt : array
        optimal fractionated average cumulative BED
    """
    para = keys
    BED_matrix = np.zeros((3,len(n_list), n_samples))
    for i, n_max in enumerate(n_list):
        dose = keys.abt/(2*n_max) * (np.sqrt(n_max ** 2 + 4*n_max*keys.tumor_goal/keys.abt) - n_max)
        para.number_of_fractions = n_max
        for j in range(n_samples):
            sf_list = np.random.normal(para.fixed_mean, para.fixed_std, n_max+1)
            para.sparing_factors = sf_list
            # calculate uniform fractionation
            BED_matrix[0][i][j] = np.sum(bed_calc0(dose, para.abn, sf_list[1:]))
            # calculate adaptive fractionation
            BED_matrix[1][i][j] = afx.multiple('oar', para).oar_sum
            # calculate optimal fractionation if all sparing factors are known at the beginning
            def bed_calc_d(dose):
                cum_bed = afx.bed_calc0(dose, para.abn, sf_list[1:])
                return np.sum(cum_bed)
            d_in = dose * np.ones(n_max)
            # define tumor BED constraint
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(afx.bed_calc0(x, para.abt)) - para.tumor_goal}]
            BED_matrix[2][i][j] = opt.minimize(bed_calc_d, x0=d_in, constraints=cons).fun

        BED_means = np.mean(BED_matrix, axis=2)
        BED_uft, BED_aft, BED_opt = BED_means[0], BED_means[1], BED_means[2]
    return BED_uft, BED_aft, BED_opt


def c_calc_0(keys):
    """
    For a specified n_pres gives the optimal C,
    when minimising OAR BED 

    Parameters
    ----------
    keys : dict
        algorithm instructions.

    Returns
    -------
    c : positive float
        optimal parameter for achieving n_pres fractions.
    """
    n_upper = keys.number_of_fractions
    n_list = np.arange(1, n_upper+1)

    def cost_fit_func(a, b):
        """
        Fit function for total OAR BED cost,
        derived from optimal fraction decision-making

        Parameters
        ----------
        a, b : float
            fit parameter.

        Returns
        -------
        cost : positive float
            total OAR BED
        """
        curvature = 4 * keys.tumor_goal / keys.abt
        cost = a * (n_list - np.sqrt(n_list**2 + curvature * n_list)) + b
        return cost
    
    _, y, _ = cost_func(keys, n_list, 15)
    
    popt, _ = opt.curve_fit(cost_fit_func, n_list, y)
    
    # popt = opt.minimize(cost_fit_func, x0=[10,100])
    return popt

