# -*- coding: utf-8 -*-
"""
2D state space (tracking sparing factor and tumor BED).
In this function the optimal fraction doses are compueted
based on a prescribed tumor dose while minimising OAR BED.
"""

import numpy as np
from scipy.interpolate import interp1d
from adaptfx import std_calc, truncated_normal, sf_probdist, bed_calc_matrix, bed_calc0, convert_to_physical, SETTING_DICT, DotDict

BED_calc_matrix = bed_calc_matrix
BED_calc0 = bed_calc0

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


def min_oar_old(keys, sets=SETTING_DICT):
    fraction = keys.fraction
    number_of_fractions = keys.number_of_fractions
    accumulated_tumor_dose = keys.accumulated_tumor_dose
    sparing_factors_public = keys.sparing_factors_public
    alpha = keys.alpha
    beta = keys.beta
    tumor_goal = keys.tumor_goal
    abt = keys.abt
    abn = keys.abn
    min_dose = keys.min_dose
    max_dose = keys.max_dose
    fixed_prob = keys.fixed_prob
    fixed_mean = keys.fixed_mean
    fixed_std = keys.fixed_std
    # ---------------------------------------------------------------------- #
    # check in which fraction policy should be returned
    policy_plot = 1 if sets.plot_policy == fraction else 0

    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors_public
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_calc(sparing_factors_public, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = truncated_normal(mean=mean, std=standard_deviation, low=0.01, upp=1.3)
    bedt_states = np.arange(accumulated_tumor_dose, tumor_goal, 1)
    bedt_states = np.concatenate(
        (bedt_states, [tumor_goal, tumor_goal + 1])
    )  # add an extra step outside of our prescribed tumor dose which will be penalised to make sure that we aim at the prescribe tumor dose
    [sf, prob] = sf_probdist(X, sets.sf_low, sets.sf_high,
        sets.sf_stepsize, sets.sf_prob_threshold)
    # we prepare an empty values list and open an action space which is equal to all the dose numbers that can be given in one fraction
    values = np.zeros(
        (number_of_fractions - fraction, len(bedt_states), len(sf))
    )  # 2d values list with first indice being the BED and second being the sf
    max_physical_dose = convert_to_physical(tumor_goal, abt)
    if max_dose == -1:
        max_dose = max_physical_dose
    elif max_dose > max_physical_dose:
        # if the max dose is too large we lower it, so we dont needlessly check too many actions
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - 0.1
    actionspace = np.arange(min_dose, max_dose + 0.1, 0.1)
    # now we set up the policy array which has len(BEDT)*len(sf)*len(actionspace) entries. We give each action the same probability to start with
    policy = np.zeros((number_of_fractions - fraction, len(bedt_states), len(sf)))

    for state, fraction_state in enumerate(
        np.arange(number_of_fractions, fraction - 1, -1)
    ):  # We have five fractionations with 2 special cases 0 and 4
        if (
            state == number_of_fractions - 1
        ):  # first state with no prior dose delivered so we dont loop through BEDT
            bedn = BED_calc_matrix(
                sparing_factors_public[-1], abn, actionspace
            )  # calculate all delivered doses to the Normal tissues (the penalty)
            future_values_func = interp1d(bedt_states, (values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                BED_calc0(actionspace, abt)
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            vs = -bedn + future_values
            policy4 = vs.argmax(axis=1)
        elif (
            fraction_state == fraction and fraction != number_of_fractions
        ):  # actual fraction
            actionspace_clipped = actionspace[
                0 : max_action(accumulated_tumor_dose, actionspace, tumor_goal) + 1
            ]
            bedn = BED_calc_matrix(sparing_factors_public[-1], abn, actionspace_clipped)
            future_bedt = accumulated_tumor_dose + BED_calc0(actionspace_clipped, abt)
            future_bedt[future_bedt > tumor_goal] = tumor_goal + 1
            penalties = np.zeros(future_bedt.shape)
            penalties[future_bedt > tumor_goal] = -10000
            future_values_func = interp1d(bedt_states, (values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_bedt
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            vs = -bedn + future_values + penalties
            policy4 = vs.argmax(axis=1)
        elif (
            fraction == number_of_fractions
        ):  # in this state no penalty has to be defined as the value is not relevant
            best_action = convert_to_physical(tumor_goal-accumulated_tumor_dose, abt)
            if accumulated_tumor_dose > tumor_goal:
                best_action = 0
            if best_action < min_dose:
                best_action = min_dose
            if best_action > max_dose:
                best_action = max_dose
            last_BEDN = BED_calc0(best_action, abn, sparing_factors_public[-1])
            policy4 = best_action * 10
        else:
            future_value_prob = (values[state - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt_states, future_value_prob)
            for tumor_index, tumor_value in enumerate(
                bedt_states
            ):  # this and the next for loop allow us to loop through all states
                actionspace_clipped = actionspace[
                    0 : max_action(tumor_value, actionspace, tumor_goal) + 1
                ]  # we only allow the actions that do not overshoot
                bedn = BED_calc_matrix(
                    sf, abn, actionspace_clipped
                )  # this one could be done outside of the loop and only the clipping would happen inside the loop.
                bed = BED_calc_matrix(np.ones(len(sf)), abt, actionspace_clipped)
                if state != 0:
                    future_bedt = tumor_value + bed
                    future_bedt[future_bedt > tumor_goal] = tumor_goal + 1
                    future_values = future_values_func(
                        future_bedt
                    )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                    penalties = np.zeros(future_bedt.shape)
                    penalties[future_bedt > tumor_goal] = -10000
                    vs = -bedn + future_values + penalties
                    if vs.size == 0:
                        best_action = np.zeros(len(sf))
                        valer = np.zeros(len(sf))
                    else:
                        best_action = vs.argmax(axis=1)
                        valer = vs.max(axis=1)
                else:  # last state no more further values to add
                    best_action = convert_to_physical(tumor_goal-tumor_value, abt)
                    if best_action > max_dose:
                        best_action = max_dose
                    if best_action < min_dose:
                        best_action = min_dose
                    last_BEDN = BED_calc0(best_action, abn, sf)
                    future_bedt = tumor_value + BED_calc0(best_action, abt)
                    underdose_penalty = 0
                    overdose_penalty = 0
                    if future_bedt < tumor_goal:
                        underdose_penalty = (future_bedt - tumor_goal) * 10
                    if future_bedt > tumor_goal:
                        overdose_penalty = -10000
                    valer = (
                        -last_BEDN
                        + underdose_penalty * np.ones(sf.shape)
                        + overdose_penalty * np.ones(sf.shape)
                    )  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for

                policy[state][tumor_index] = best_action
                values[state][tumor_index] = valer
    if fraction != number_of_fractions:
        optimal_action = actionspace[policy4]
    if fraction == number_of_fractions:
        optimal_action = policy4 / 10
    tumor_dose = BED_calc0(optimal_action, abt)
    oar_dose = BED_calc0(optimal_action, abn, sparing_factors_public[-1])
    output = DotDict({'physical_dose': optimal_action, 'tumor_dose': tumor_dose, 
        'oar_dose': oar_dose, 'sf': sf, 'states': bedt_states})
    if policy_plot:
        output['policy'] = policy
    return output