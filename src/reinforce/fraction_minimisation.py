# -*- coding: utf-8 -*-
"""
Interpolation module for number of fraction minimisation
with minimum and maximum physical dose per fraction (to tumor).
2D state space (tracking sparing factor and tumor BED).
In this program the optimal fraction doses are compueted
based on a prescribed tumor dose while minimizing the
number of fractions used for the treatment.

whole_plan computes the doses for a whole
treatment plan (when all sparing factors are known).

whole_plan computes the doses for a whole
treatment plan (when all sparing factors are known).
"""

import numpy as np
from scipy.interpolate import interp1d
from common.maths import std_calc, get_truncated_normal, probdist
from common.radiobiology import BED_calc_matrix, BED_calc0, max_action, convert_to_physical

def value_eval(
    fraction,
    number_of_fractions,
    accumulated_dose,
    sparing_factors,
    alpha,
    beta,
    goal,
    abt,
    abn,
    C,
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
        Number of the actual fraction.
    number_of_fractions : integer
        number of fractions that will be delivered.
    accumulated_dose : float
        accumulated tumor BED.
    sparing_factors : list/array
        list or array of all observed sparing factors. Include planning
        session sparing factor!.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients
    goal : float
        prescribed tumor BED.
    abt : float, optional
        alpha beta ratio of the tumor. The default is 10.
    abn : float, optional
        alpha beta ratio of the organ at risk. The default is 3.
    C: float
        fixed constant to penalize for each additional fraction that is used
    min_dose : float
        minimal physical doses to be delivered in one fraction.
        The doses are aimed at PTV 95.
    max_dose : float
        maximal physical doses to be delivered in one fraction.
        The doses are aimed at PTV 95.
    fixed_prob : int
        this variable is to turn on a fixed probability distribution.
        If the variable is not used (0), then the probability will be updated.
        If the variable is turned to (1), the inserted mean and std will be used
        for a fixed sparing factor distribution. Then alpha and beta are unused.
    fixed_mean: float
        mean of the fixed sparing factor normal distribution.
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution.

    Returns
    -------
    list
        Returns list with policies, relevant sparing factors range,
        physical dose to be delivered, tumor BED to be delivered,
        OAR BED to be delivered.

    """
    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_calc(sparing_factors, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = get_truncated_normal(mean=mean, sd=standard_deviation, low=0.01, upp=1.3)
    BEDT = np.arange(accumulated_dose, goal, 1)
    BEDT = np.concatenate(
        (BEDT, [goal, goal + 1])
    )  # add an extra step outside of our prescribed tumor dose which will be penalized to make sure that we aim at the prescribe tumor dose
    prob = np.array(probdist(X))
    sf = np.arange(0.01, 1.71, 0.01)
    sf = sf[prob > 0.00001]
    prob = prob[prob > 0.00001]
    # we prepare an empty values list and open an action space which is equal to all the dose numbers that can be given in one fraction
    Values = np.zeros(
        (number_of_fractions - fraction, len(BEDT), len(sf))
    )  # 2d values list with first indice being the BED and second being the sf
    max_physical_dose = convert_to_physical(goal, abt)
    if max_dose > max_physical_dose:  
        # if the max dose is too large we lower it, so we dont needlessly check too many actions
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - 0.1
    actionspace = np.arange(min_dose, max_dose + 0.1, 0.1)
    # now we set up the policy array which has len(BEDT)*len(sf)*len(actionspace) entries. We give each action the same probability to start with
    policy = np.zeros((number_of_fractions - fraction, len(BEDT), len(sf)))

    for state, fraction_state in enumerate(
        np.arange(number_of_fractions, fraction -1, -1)
    ):
        if (
            state == number_of_fractions - 1 #fraction_state == 1
        ):  # first state with no prior dose delivered so we dont loop through BEDT
            BEDN = BED_calc_matrix(
                sparing_factors[-1], abn, actionspace
            )  # calculate all delivered doses to the Normal tissues (the penalty)
            future_BEDT = BED_calc0(actionspace, abt)
            future_values_func = interp1d(BEDT, (Values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_BEDT
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            C_penalties = np.zeros(future_BEDT.shape)
            C_penalties[future_BEDT < goal] = -C
            Vs = -BEDN + C_penalties + future_values
            policy4 = Vs.argmax(axis=1)
        elif (
            accumulated_dose >= goal
        ):
            best_action = 0
            last_BEDN = BED_calc0(best_action, abn, sparing_factors[-1])
            policy4 = 0
            break
        elif (
            fraction_state == fraction and fraction != number_of_fractions
        ):  # actual fraction is first state to calculate
            actionspace_clipped = actionspace[
                0 : max_action(accumulated_dose, actionspace, goal) + 1
            ]
            BEDN = BED_calc_matrix(sparing_factors[-1], abn, actionspace_clipped)
            future_BEDT = accumulated_dose + BED_calc0(actionspace_clipped, abt)
            future_BEDT[future_BEDT > goal] = goal + 1
            penalties = np.zeros(future_BEDT.shape)
            C_penalties = np.zeros(future_BEDT.shape)
            penalties[future_BEDT > goal] = 0 #overdosing is indirectly penalised with BEDN
            C_penalties[future_BEDT < goal] = -C
            future_values_func = interp1d(BEDT, (Values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_BEDT
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            Vs = -BEDN + C_penalties + future_values + penalties
            policy4 = Vs.argmax(axis=1)
        elif (
            fraction == number_of_fractions
        ):  # in this state no penalty has to be defined as the value is not relevant
            best_action = convert_to_physical(goal-accumulated_dose, abt)
            if accumulated_dose > goal:
                best_action = 0
            if best_action < min_dose:
                best_action = min_dose
            if best_action > max_dose:
                best_action = max_dose
            last_BEDN = BED_calc0(best_action, abn, sparing_factors[-1])
            policy4 = best_action * 10
        else:
            future_value_prob = (Values[state - 1] * prob).sum(axis=1)
            future_values_func = interp1d(BEDT, future_value_prob)
            for tumor_index, tumor_value in enumerate(
                BEDT
            ):  # this and the next for loop allow us to loop through all states
                actionspace_clipped = actionspace[
                    0 : max_action(tumor_value, actionspace, goal) + 1
                ]  # we only allow the actions that do not overshoot
                BEDN = BED_calc_matrix(
                    sf, abn, actionspace_clipped
                )  # this one could be done outside of the loop and only the clipping would happen inside the loop.
                BED = BED_calc_matrix(np.ones(len(sf)), abt, actionspace_clipped)
                if state != 0 and tumor_value < goal:
                    future_BEDT = tumor_value + BED
                    future_BEDT[future_BEDT > goal] = goal + 1
                    future_values = future_values_func(
                        future_BEDT
                    )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                    C_penalties = np.zeros(future_BEDT.shape)
                    C_penalties[future_BEDT < goal] = -C
                    Vs = -BEDN + C_penalties + future_values
                    if Vs.size == 0:
                        best_action = np.zeros(len(sf))
                        valer = np.zeros(len(sf))
                    else:
                        best_action = actionspace_clipped[Vs.argmax(axis=1)]
                        valer = Vs.max(axis=1)
                elif tumor_value > goal: #calculate value for terminal case
                    best_action = 0
                    future_BEDT = tumor_value
                    valer = (
                        + np.zeros(sf.shape)
                    )  
                else:  # last state no more further values to add
                    best_action = convert_to_physical(goal-tumor_value, abt)
                    if best_action > max_dose:
                        best_action = max_dose
                    if best_action < min_dose:
                        best_action = min_dose
                    last_BEDN = BED_calc0(best_action, abn, sf)
                    future_BEDT = tumor_value + BED_calc0(best_action, abt)
                    underdose_penalty = 0
                    overdose_penalty = 0
                    if future_BEDT < goal:
                        underdose_penalty = -10000
                    if future_BEDT > goal:
                        overdose_penalty = 0
                    valer = (
                        -last_BEDN
                        + underdose_penalty * np.ones(sf.shape)
                        + overdose_penalty * np.ones(sf.shape)
                    )  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for

                policy[state][tumor_index] = best_action
                Values[state][tumor_index] = valer
    if fraction != number_of_fractions:
        physical_dose = actionspace[policy4]
    if fraction == number_of_fractions:
        physical_dose = policy4 / 10
    tumor_dose = BED_calc0(physical_dose, abt)
    OAR_dose = BED_calc0(physical_dose, abn, sparing_factors[-1])
    return [physical_dose, tumor_dose, OAR_dose]
    # return [physical_dose, tumor_dose, OAR_dose, policy, sf, BEDT]