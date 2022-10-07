# -*- coding: utf-8 -*-
"""
2D state space (tracking sparing factor and OAR BED).
In this function the optimal fraction doses are computed 
based on a maximal OAR dose while maximising tumor BED.
"""

import numpy as np
from scipy.interpolate import interp2d
from common.maths import std_calc, get_truncated_normal, probdist
from common.radiobiology import argfind, BED_calc_matrix, BED_calc0, convert_to_physical

def value_eval(
    fraction,
    number_of_fractions,
    accumulated_oar_dose,
    sparing_factors,
    alpha,
    beta,
    oar_limit,
    abt,
    abn,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    calculates the optimal dose for the desired fraction.

    Parameters
    ----------
    fraction : integer
        number of actual fraction.
    number_of_fractions : integer
        number of fractions that will be delivered.
    accumulated_oar_dose : float
        accumulated BED in OAR (from previous fractions)
    sparing_factors : list/array
        list of sparing factor distribution.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    oar_limit : float
        upper BED limit of OAR.
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
    min_dose : float
        minimal physical doses to be delivered in one fraction.
        The doses are aimed at PTV 95.
    max_dose : float
        maximal physical doses to be delivered in one fraction.
        The doses are aimed at PTV 95 .
    fixed_prob : int
        this variable is to turn on a fixed probability distribution.
        If the variable is not used (0), then the probability will
        be updated. If the variable is turned to 1, the inserted mean
        and std will be used for a fixed sparing factor distribution.
        Then alpha and beta are unused.
    fixed_mean: float
        mean of the fixed sparing factor normal distribution.
    fixed_std: float
        standard deviation of the fixed sparing factor normal distribution.

    Returns
    -------
    list

    """
    
    actual_sparing = sparing_factors[-1]
    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_calc(sparing_factors, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = get_truncated_normal(mean=mean, sd=standard_deviation, low=0, upp=1.7)
    prob = np.array(probdist(X))
    sf = np.arange(0.01, 1.71, 0.01)
    sf = sf[prob > 0.00001]  # get rid of all probabilities below 10^-5
    prob = prob[prob > 0.00001]

    bedn = np.arange(accumulated_oar_dose, oar_limit + 1.6, 1)
    values = np.zeros(
        ((number_of_fractions - fraction), len(bedn), len(sf))
    )  # 2d values list with first indice being the BED and second being the sf
    if (
        max_dose > 22.3
    ):  # if the chosen maximum dose is too large, it gets reduced. So the algorithm doesn't needlessly check too many actions
        max_dose = 22.3
    if min_dose > max_dose:
        min_dose = max_dose - 0.1
    actionspace = np.arange(min_dose, max_dose + 0.1, 0.1)
    policy = np.zeros(((number_of_fractions - fraction), len(bedn), len(sf)))
    upperbound = oar_limit + 1

    delivered_doses = BED_calc_matrix(sf, abn, actionspace)
    bedt_reward = BED_calc_matrix(
        1, abt, actionspace
    )  # this is the reward for the dose deposited inside the tumor.
    bedt_transformed, meaningless = np.meshgrid(bedt_reward, np.zeros(len(sf)))

    for index, frac_state in enumerate(
        np.arange(fraction, number_of_fractions + 1)
    ):  # We have number_of fraction fractionations with 2 special cases 0 and number_of_fractions-1 (last first fraction)
        if (
            index == number_of_fractions - 1
        ):  # first state with no prior dose delivered so we dont loop through BEDT
            future_bed = accumulated_oar_dose + delivered_doses
            future_bed[
                future_bed > oar_limit
            ] = upperbound  # any dose surpassing the upper bound will be set to the upper bound which will be penalised strongly
            value_interpolation = interp2d(sf, bedn, values[index - 1])
            future_value = np.zeros(len(sf) * len(actionspace) * len(sf)).reshape(
                len(sf), len(actionspace), len(sf)
            )
            for actual_sf in range(0, len(sf)):
                future_value[actual_sf] = value_interpolation(sf, future_bed[actual_sf])
            future_values_prob = (future_value * prob).sum(
                axis=2
            )  # in this array are all future values multiplied with the probability of getting there. shape = sparing factors x actionspace
            penalties = np.zeros(future_bed.shape)
            penalties[
                future_bed > oar_limit
            ] = (
                -1000
            )  # penalising in each fraction is needed. If not, once the algorithm reached the upper bound, it would just deliver maximum dose over and over again
            vs = future_values_prob + bedt_transformed + penalties

            actual_policy = vs.argmax(axis=1)
            actual_value = vs.max(axis=1)

        else:
            if (
                index == number_of_fractions - fraction
            ):  # if we are in the actual fraction we do not need to check all possible BED states but only the one we are in
                if fraction != number_of_fractions:
                    future_bed = accumulated_oar_dose + delivered_doses
                    overdosing = (future_bed - oar_limit).clip(min=0)
                    penalties_overdose = (
                        overdosing * -1000
                    )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                    future_bed[
                        future_bed > oar_limit
                    ] = upperbound  # any dose surpassing the upper bound will be set to the upper bound which will be penalised strongly
                    value_interpolation = interp2d(
                        sf, bedn, values[index - 1]
                    )
                    future_value = np.zeros(
                        len(sf) * len(actionspace) * len(sf)
                    ).reshape(len(sf), len(actionspace), len(sf))
                    for actual_sf in range(0, len(sf)):
                        future_value[actual_sf] = value_interpolation(
                            sf, future_bed[actual_sf]
                        )
                    future_values_prob = (future_value * prob).sum(
                        axis=2
                    )  # in this array are all future values multiplied with the probability of getting there. shape = sparing factors x actionspace
                    penalties = np.zeros(future_bed.shape)
                    penalties[
                        future_bed > oar_limit
                    ] = (
                        -1000
                    )  # penalising in each fraction is needed. If not, once the algorithm reached the upper bound, it would just deliver maximum dose over and over again
                    vs = (
                        future_values_prob
                        + bedt_transformed
                        + penalties
                        + penalties_overdose
                    )
                    actual_policy = vs.argmax(axis=1)
                    actual_value = vs.max(axis=1)
                else:
                    best_action = convert_to_physical(oar_limit-accumulated_oar_dose, abn, sf)
                    if accumulated_oar_dose > oar_limit:
                        best_action = np.ones(best_action.shape) * min_dose
                    best_action[best_action < min_dose] = min_dose
                    best_action[best_action > max_dose] = max_dose
                    actual_policy = best_action * 10
                    actual_value = BED_calc0(
                        best_action, abt
                    )  # we do not need to penalise, as this value is not relevant.
            else:
                for bed_index, bed_value in enumerate(
                    bedn
                ):  # this and the next for loop allow us to loop through all states
                    future_bed = delivered_doses + bed_value
                    overdosing = (future_bed - oar_limit).clip(min=0)
                    future_bed[
                        future_bed > oar_limit
                    ] = upperbound  # any dose surpassing 90.1 is set to 90.1
                    if index == 0:  # last state no more further values to add
                        best_action = convert_to_physical(oar_limit-bed_value, abn, sf)
                        best_action[best_action < min_dose] = min_dose
                        best_action[best_action > max_dose] = max_dose
                        future_bed = BED_calc0(sf, abn, best_action) + bed_value
                        overdosing = (future_bed - oar_limit).clip(min=0)
                        penalties_overdose = (
                            overdosing * -1000
                        )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                        future_bed[
                            future_bed > oar_limit + 0.0001
                        ] = upperbound  # 0.0001 is added due to some rounding problems
                        penalties = np.zeros(future_bed.shape)
                        if bed_value < oar_limit:
                            penalties[future_bed == upperbound] = -1000
                        values[index][bed_index] = (
                            BED_calc0(best_action, abt) + penalties + penalties_overdose
                        )
                        policy[index][bed_index] = (best_action - min_dose) * 10
                    else:
                        penalties = np.zeros(future_bed.shape)
                        penalties[future_bed == upperbound] = -1000
                        penalties_overdose = (
                            overdosing * -1000
                        )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                        value_interpolation = interp2d(
                            sf, bedn, values[index - 1]
                        )
                        future_value = np.zeros((len(sf), len(actionspace), len(sf)))
                        for actual_sf in range(0, len(sf)):
                            future_value[actual_sf] = value_interpolation(
                                sf, future_bed[actual_sf]
                            )
                        future_values_prob = (future_value * prob).sum(axis=2)
                        vs = (
                            future_values_prob
                            + bedt_transformed
                            + penalties
                            + penalties_overdose
                        )
                        best_action = vs.argmax(axis=1)
                        valer = vs.max(axis=1)
                        policy[index][bed_index] = best_action
                        values[index][bed_index] = valer
    index_sf = argfind(sf, actual_sparing)
    if fraction != number_of_fractions:
        dose_delivered_tumor = BED_calc0(actionspace[actual_policy[index_sf]], abt)
        dose_delivered_oar = BED_calc0(
            actionspace[actual_policy[index_sf]], abn, actual_sparing
        )
        total_dose_delivered_oar = dose_delivered_oar + accumulated_oar_dose
        actual_dose_delivered = actionspace[actual_policy[index_sf]]
    else:
        dose_delivered_tumor = BED_calc0(actual_policy[index_sf] / 10, abt)
        dose_delivered_oar = BED_calc0(
            actual_policy[index_sf] / 10, abn, actual_sparing
        )
        total_dose_delivered_oar = dose_delivered_oar + accumulated_oar_dose
        actual_dose_delivered = actual_policy[index_sf] / 10
    return [actual_dose_delivered, dose_delivered_tumor, dose_delivered_oar]
    # return [
    #     Values,
    #     policy,
    #     actual_value,
    #     actual_policy,
    #     dose_delivered_oar,
    #     dose_delivered_tumor,
    #     total_dose_delivered_oar,
    #     actual_dose_delivered,
    # ]