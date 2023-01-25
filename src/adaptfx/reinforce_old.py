# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from adaptfx import std_posterior, truncated_normal, sf_probdist, bed_calc_matrix, bed_calc0, convert_to_physical, SETTING_DICT, DotDict

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


def min_oar_bed_old(keys, sets=SETTING_DICT):
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
        standard_deviation = std_posterior(sparing_factors_public, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = truncated_normal(mean, standard_deviation, sets.sf_low, sets.sf_high)
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

    output = DotDict({})

    output.physical_dose = optimal_action
    output.tumor_dose = tumor_dose
    output.oar_dose = oar_dose
    
    return output






def min_n_frac_old(keys, sets=SETTING_DICT):
    fraction = keys.fraction
    number_of_fractions = keys.number_of_fractions
    accumulated_tumor_dose = keys.accumulated_tumor_dose
    sparing_factors_public = keys.sparing_factors_public
    alpha = keys.alpha
    beta = keys.beta
    tumor_goal = keys.tumor_goal
    c = keys.c
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
        standard_deviation = std_posterior(sparing_factors_public, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = truncated_normal(mean, standard_deviation, sets.sf_low, sets.sf_high)
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
        np.arange(number_of_fractions, fraction -1, -1)
    ):
        if (
            state == number_of_fractions - 1 #fraction_state == 1
        ):  # first state with no prior dose delivered so we dont loop through BEDT
            bedn = bed_calc_matrix(
                sparing_factors_public[-1], abn, actionspace
            )  # calculate all delivered doses to the Normal tissues (the penalty)
            future_bedt = bed_calc0(actionspace, abt)
            future_values_func = interp1d(bedt_states, (values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_bedt
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            c_penalties = np.zeros(future_bedt.shape)
            c_penalties[future_bedt < tumor_goal] = -c
            vs = -bedn + c_penalties + future_values
            policy4 = vs.argmax(axis=1)
        elif (
            accumulated_tumor_dose >= tumor_goal
        ):
            best_action = 0
            last_BEDN = bed_calc0(best_action, abn, sparing_factors_public[-1])
            policy4 = 0
            break
        elif (
            fraction_state == fraction and fraction != number_of_fractions
        ):  # actual fraction is first state to calculate
            actionspace_clipped = actionspace[
                0 : max_action(accumulated_tumor_dose, actionspace, tumor_goal) + 1
            ]
            bedn = bed_calc_matrix(sparing_factors_public[-1], abn, actionspace_clipped)
            future_bedt = accumulated_tumor_dose + bed_calc0(actionspace_clipped, abt)
            future_bedt[future_bedt > tumor_goal] = tumor_goal + 1
            penalties = np.zeros(future_bedt.shape)
            c_penalties = np.zeros(future_bedt.shape)
            penalties[future_bedt > tumor_goal] = 0 #overdosing is indirectly penalised with BEDN
            c_penalties[future_bedt < tumor_goal] = -c
            future_values_func = interp1d(bedt_states, (values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_bedt
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            vs = -bedn + c_penalties + future_values + penalties
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
            last_BEDN = bed_calc0(best_action, abn, sparing_factors_public[-1])
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
                bedn = bed_calc_matrix(
                    sf, abn, actionspace_clipped
                )  # this one could be done outside of the loop and only the clipping would happen inside the loop.
                bed = bed_calc_matrix(np.ones(len(sf)), abt, actionspace_clipped)
                if state != 0 and tumor_value < tumor_goal:
                    future_bedt = tumor_value + bed
                    future_bedt[future_bedt > tumor_goal] = tumor_goal + 1
                    future_values = future_values_func(
                        future_bedt
                    )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                    c_penalties = np.zeros(future_bedt.shape)
                    c_penalties[future_bedt < tumor_goal] = -c
                    vs = -bedn + c_penalties + future_values
                    if vs.size == 0:
                        best_action = np.zeros(len(sf))
                        valer = np.zeros(len(sf))
                    else:
                        best_action = actionspace_clipped[vs.argmax(axis=1)]
                        valer = vs.max(axis=1)
                elif tumor_value > tumor_goal: #calculate value for terminal case
                    best_action = 0
                    future_bedt = tumor_value
                    valer = (
                        + np.zeros(sf.shape)
                    )  
                else:  # last state no more further values to add
                    best_action = convert_to_physical(tumor_goal-tumor_value, abt)
                    if best_action > max_dose:
                        best_action = max_dose
                    if best_action < min_dose:
                        best_action = min_dose
                    last_BEDN = bed_calc0(best_action, abn, sf)
                    future_bedt = tumor_value + bed_calc0(best_action, abt)
                    underdose_penalty = 0
                    overdose_penalty = 0
                    if future_bedt < tumor_goal:
                        underdose_penalty = -10000
                    if future_bedt > tumor_goal:
                        overdose_penalty = 0
                    valer = (
                        -last_BEDN
                        + underdose_penalty * np.ones(sf.shape)
                        + overdose_penalty * np.ones(sf.shape)
                    )  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for

                policy[state][tumor_index] = best_action
                values[state][tumor_index] = valer

    if sets.plot_policy or sets.plot_value:
        policy = np.concatenate([np.zeros((1, len(bedt_states), len(sf))), policy[::-1]])
        values = np.concatenate([np.zeros((1, len(bedt_states), len(sf))), values[::-1]])
    if fraction != number_of_fractions:
        optimal_action = actionspace[policy4]
    if fraction == number_of_fractions:
        optimal_action = policy4 / 10
    tumor_dose = bed_calc0(optimal_action, abt)
    oar_dose = bed_calc0(optimal_action, abn, sparing_factors_public[-1])

    output = DotDict({})

    output.physical_dose = optimal_action
    output.tumor_dose = tumor_dose
    output.oar_dose = oar_dose
    
    return output







def max_tumor_bed_old(keys, sets=SETTING_DICT):
    fraction = keys.fraction
    number_of_fractions = keys.number_of_fractions
    accumulated_oar_dose = keys.accumulated_tumor_dose
    sparing_factors_public = keys.sparing_factors_public
    alpha = keys.alpha
    beta = keys.beta
    oar_limit = keys.oar_limit
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
    
    actual_sparing = sparing_factors_public[-1]
    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors_public
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_posterior(sparing_factors_public, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = truncated_normal(mean, standard_deviation, sets.sf_low, sets.sf_high)
    [sf, prob] = sf_probdist(X, sets.sf_low, sets.sf_high,
        sets.sf_stepsize, sets.sf_prob_threshold)

    bedn = np.arange(accumulated_oar_dose, oar_limit + 1.6, 1)
    values = np.zeros(
        ((number_of_fractions - fraction), len(bedn), len(sf))
    )  # 2d values list with first indice being the BED and second being the sf
    max_physical_dose = convert_to_physical(oar_limit, abn, actual_sparing)
    if max_dose == -1:
        max_dose = max_physical_dose
    elif max_dose > max_physical_dose:
        # if the max dose is too large we lower it, so we dont needlessly check too many actions
        max_dose = max_physical_dose
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

    output = DotDict({})

    output.physical_dose = actual_dose_delivered
    output.tumor_dose = dose_delivered_tumor
    output.oar_dose = dose_delivered_oar
    
    return output






def min_oar_max_tumor_old(keys, sets=SETTING_DICT):
    fraction = keys.fraction
    number_of_fractions = keys.number_of_fractions
    accumulated_tumor_dose = keys.accumulated_tumor_dose
    accumulated_oar_dose = keys.accumulated_oar_dose
    sparing_factors_public = keys.sparing_factors_public
    alpha = keys.alpha
    beta = keys.beta
    tumor_goal = keys.tumor_goal
    oar_limit = keys.oar_limit
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
        standard_deviation = std_posterior(sparing_factors_public, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = truncated_normal(mean, standard_deviation, sets.sf_low, sets.sf_high)
    [sf, prob] = sf_probdist(X, sets.sf_low, sets.sf_high,
        sets.sf_stepsize, sets.sf_prob_threshold)
    underdosepenalty = 10
    print(accumulated_oar_dose, accumulated_tumor_dose)
    bedt = np.arange(accumulated_tumor_dose, tumor_goal, 1)  # tumordose
    bedn = np.arange(accumulated_oar_dose, oar_limit, 1)  # OAR dose
    bedn = np.concatenate((bedn, [oar_limit, oar_limit + 1]))
    bedt = np.concatenate((bedt, [tumor_goal, tumor_goal + 1]))
    values = np.zeros(
        [(number_of_fractions - fraction), len(bedt), len(bedn), len(sf)]
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
    policy = np.zeros(
        ((number_of_fractions - fraction), len(bedt), len(bedn), len(sf))
    )
    upperbound_normal_tissue = oar_limit + 1
    upperbound_tumor = tumor_goal + 1

    oar_dose = BED_calc_matrix(
        sf, abn, actionspace
    )  # calculates the dose that is deposited into the normal tissue for all sparing factors
    tumor_dose = BED_calc_matrix(1, abt, actionspace)[
        0
    ]  # this is the dose delivered to the tumor
    actual_fraction_sf = argfind(sf, sparing_factors_public[-1])

    for index, frac_state_plus in enumerate(
        np.arange(number_of_fractions + 1, fraction, -1)
    ):  # We have five fractionations with 2 special cases 0 and 4
        frac_state = frac_state_plus - 1
        if (
            frac_state == 1
        ):  # first state with no prior dose delivered so we dont loop through BEDNT
            future_oar = accumulated_oar_dose + oar_dose[actual_fraction_sf]
            future_tumor = accumulated_tumor_dose + tumor_dose
            future_oar[
                future_oar > oar_limit
            ] = upperbound_normal_tissue  # any dose surpassing the upper bound will be set to the upper bound which will be penalised strongly
            future_tumor[future_tumor > tumor_goal] = upperbound_tumor
            future_values_prob = (values[index - 1] * prob).sum(
                axis=2
            )  # future values of tumor and oar state
            value_interpolation = RegularGridInterpolator(
                (bedt, bedn), future_values_prob
            )
            future_value_actual = value_interpolation(
                np.array([future_tumor, future_oar]).T
            )
            vs = future_value_actual - oar_dose[actual_fraction_sf]
            actual_policy = vs.argmax(axis=0)

        elif (
            frac_state == fraction
        ):  # if we are in the actual fraction we do not need to check all possible BED states but only the one we are in
            if fraction != number_of_fractions:
                future_oar = accumulated_oar_dose + oar_dose[actual_fraction_sf]
                overdosing = (future_oar - oar_limit).clip(min=0)
                future_oar[
                    future_oar > oar_limit
                ] = upperbound_normal_tissue  # any dose surpassing the upper bound will be set to the upper bound which will be penalised strongly
                future_tumor = accumulated_tumor_dose + tumor_dose
                future_tumor[future_tumor > tumor_goal] = upperbound_tumor
                future_values_prob = (values[index - 1] * prob).sum(
                    axis=2
                )  # future values of tumor and oar state
                value_interpolation = RegularGridInterpolator(
                    (bedt, bedn), future_values_prob
                )
                penalties = (
                    overdosing * -10000000000
                )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                future_value_actual = value_interpolation(
                    np.array([future_tumor, future_oar]).T
                )
                vs = future_value_actual - oar_dose[actual_fraction_sf] + penalties
                actual_policy = vs.argmax(axis=0)
            else:
                sf_end = sparing_factors_public[-1]
                best_action_oar = convert_to_physical(oar_limit-accumulated_oar_dose, abn, sf_end)
                best_action_tumor = convert_to_physical(tumor_goal-accumulated_tumor_dose, abt)
                best_action = np.min([best_action_oar, best_action_tumor], axis=0)
                if accumulated_oar_dose > oar_limit or accumulated_tumor_dose > tumor_goal:
                    best_action = np.ones(best_action.shape) * min_dose
                if best_action > max_dose:
                    best_action = max_dose
                if best_action < min_dose:
                    best_action = min_dose

                future_tumor = accumulated_tumor_dose + BED_calc0(best_action, abt)
                future_oar = accumulated_oar_dose + BED_calc0(best_action, abn, sf_end)
                actual_policy = best_action * 10
        else:
            future_values_prob = (values[index - 1] * prob).sum(
                axis=2
            )  # future values of tumor and oar state
            value_interpolation = RegularGridInterpolator(
                (bedt, bedn), future_values_prob
            )  # interpolation function
            for tumor_index, tumor_value in enumerate(bedt):
                for oar_index, oar_value in enumerate(
                    bedn
                ):  # this and the next for loop allow us to loop through all states
                    future_oar = oar_dose + oar_value
                    overdosing = (future_oar - oar_limit).clip(min=0)
                    future_oar[
                        future_oar > oar_limit
                    ] = upperbound_normal_tissue  # any dose surpassing 90.1 is set to 90.1
                    future_tumor = tumor_value + tumor_dose
                    future_tumor[
                        future_tumor > tumor_goal
                    ] = upperbound_tumor  # any dose surpassing the tumor bound is set to tumor_bound + 0.1

                    if (
                        frac_state == number_of_fractions
                    ):  # last state no more further values to add
                        best_action_oar = convert_to_physical(oar_limit-oar_value, abn, sf) 
                        # calculate maximal dose that can be delivered to OAR and tumor
                        best_action_tumor = convert_to_physical(tumor_goal-tumor_value, abt, sf*0+1)
                        best_action = np.min(
                            [best_action_oar, best_action_tumor], axis=0
                        )  # take the smaller of both doses to not surpass the limit
                        best_action[best_action > max_dose] = max_dose
                        best_action[best_action < min_dose] = min_dose
                        if (
                            oar_value > oar_limit or tumor_value > tumor_goal
                        ):  # if the limit is already surpassed we add a penaltsy
                            best_action = np.ones(best_action.shape) * min_dose
                        future_oar = oar_value + BED_calc0(best_action, abn, sf)
                        future_tumor = tumor_value + BED_calc0(best_action, abt, 1)
                        overdose_penalty2 = np.zeros(
                            best_action.shape
                        )  # we need a second penalty if we overdose in the last fraction
                        overdose_penalty3 = np.zeros(best_action.shape)
                        overdose_penalty2[
                            future_tumor > tumor_goal + 0.0001
                        ] = -100000000000
                        overdose_penalty3[
                            future_oar > oar_limit + 0.0001
                        ] = (
                            -100000000000
                        )  # A small number has to be added as sometimes 90. > 90 was True
                        end_penalty = (
                            -abs(future_tumor - tumor_goal) * underdosepenalty
                        )  # the farther we are away from the prescribed dose, the higher the penalty. Under- and overdosing is punished
                        end_penalty_oar = (
                            -(future_oar - oar_limit).clip(min=0) * 1000
                        )  # if overdosing the OAR is not preventable, the overdosing should stay as low as possible
                        values[index][tumor_index][oar_index] = (
                            end_penalty
                            - BED_calc0(best_action, abn, sf)
                            + overdose_penalty2
                            + overdose_penalty3
                            + end_penalty_oar
                        )  # we also substract all the dose delivered to the OAR so the algorithm tries to minimise it
                        policy[index][tumor_index][oar_index] = best_action * 10
                    else:
                        future_value = np.zeros([len(sf), len(actionspace)])
                        for actual_sf in range(0, len(sf)):
                            future_value[actual_sf] = value_interpolation(
                                np.array([future_tumor, future_oar[actual_sf]]).T
                            )
                        penalties = (
                            overdosing * -10000000000
                        )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                        vs = future_value - oar_dose + penalties
                        best_action = vs.argmax(axis=1)
                        valer = vs.max(axis=1)
                        policy[index][tumor_index][oar_index] = best_action
                        values[index][tumor_index][oar_index] = valer
    if fraction != number_of_fractions:
        physical_dose = actionspace[actual_policy]
    else:
        physical_dose = actual_policy / 10
    tumor_dose = BED_calc0(physical_dose, abt)
    oar_dose = BED_calc0(physical_dose, abn, sparing_factors_public[-1])

    output = DotDict({})

    output.physical_dose = physical_dose
    output.tumor_dose = tumor_dose
    output.oar_dose = oar_dose
    
    return output