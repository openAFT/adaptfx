# -*- coding: utf-8 -*-
"""
This function tracks sparing factor, tumor BED and OAR BED.
If the prescribed tumor dose can be reached, the OAR dose
is minimised. If the  prescribed tumor dose can not be reached,
while staying below maximum BED, the tumor dose is maximised.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from common.maths import std_calc, get_truncated_normal, probdist
from common.radiobiology import argfind, BED_calc_matrix, BED_calc0, convert_to_physical

# right now once 90 is hit it doesnt seem to matter how much
# is overdosed. Somehow this must be fixed


def value_eval(
    fraction,
    number_of_fractions,
    accumulated_OAR_dose,
    accumulated_tumor_dose,
    sparing_factors,
    OAR_limit,
    tumor_goal,
    alpha,
    beta,
    abt,
    abn,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    Calculates the optimal dose for the desired fraction.
    
    Parameters
    ----------
    fraction : integer
        number of actual fraction.
    number_of_fractions : integer
        number of fractions that will be delivered.
    accumulated_OAR_dose : float
        accumulated BED in OAR (from previous fractions).
    accumulated_tumor_dose : float
        accumulated BED in tumor (from previous fractions).
    sparing_factors : TYPE
        list or array of all sparing factors that have been observed-
    OAR limit : float
        upper BED limit of OAR
    tumor_goal : float
        prescribed tumor BED.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
    min_dose : float
        minimal physical doses to be delivered in one fraction.
        The doses are aimed at PTV 95.
    max_dose : float
        maximal physical doses to be delivered in one fraction.
        The doses are aimed at PTV 95.
    fixed_prob : int
        this variable is to turn on a fixed probability distribution.
        If the variable is not used (0), then the probability will
        be updated. If the variable is turned to 1, the inserted mean
        and std will be used for a fixed sparing factor distribution.
        Then alpha and beta are unused.
    fixed_mean: float
        mean of the fixed sparing factor normal distribution.
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution.

    Returns
    -------
    list
        list with following arrays/values:
        physical_dose: optimal physical dose for actual fraction
        accumulated_tumor_dose: accumulated tumor BED
        accumulated_OAR_dose: accumulated OAR BED
        tumor_dose: tumor BED to be delivered
        OAR_dose: OAR BED to be delivered

    """

    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_calc(sparing_factors, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = get_truncated_normal(mean=mean, sd=standard_deviation, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf = np.arange(0.01, 1.71, 0.01)
    sf = sf[prob > 0.00001]  # get rid of all probabilities below 10^-5
    prob = prob[prob > 0.00001]
    underdosepenalty = 10
    BEDT = np.arange(accumulated_tumor_dose, tumor_goal, 1)  # tumordose
    BEDNT = np.arange(accumulated_OAR_dose, OAR_limit, 1)  # OAR dose
    BEDNT = np.concatenate((BEDNT, [OAR_limit, OAR_limit + 1]))
    BEDT = np.concatenate((BEDT, [tumor_goal, tumor_goal + 1]))
    Values = np.zeros(
        [(number_of_fractions - fraction), len(BEDT), len(BEDNT), len(sf)]
    )  # 2d values list with first indice being the BED and second being the sf
    max_physical_dose = convert_to_physical(tumor_goal, abt)
    if max_dose > max_physical_dose:  
        # we constrain the maximum dose so that no more dose than what is needed would be checked in the actionspace
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - 0.1
    actionspace = np.arange(min_dose, max_dose + 0.1, 0.1)
    policy = np.zeros(
        ((number_of_fractions - fraction), len(BEDT), len(BEDNT), len(sf))
    )
    upperbound_normal_tissue = OAR_limit + 1
    upperbound_tumor = tumor_goal + 1

    OAR_dose = BED_calc_matrix(
        sf, abn, actionspace
    )  # calculates the dose that is deposited into the normal tissue for all sparing factors
    tumor_dose = BED_calc_matrix(1, abt, actionspace)[
        0
    ]  # this is the dose delivered to the tumor
    actual_fraction_sf = argfind(sf, sparing_factors[-1])

    for index, frac_state_plus in enumerate(
        np.arange(number_of_fractions + 1, fraction, -1)
    ):  # We have five fractionations with 2 special cases 0 and 4
        frac_state = frac_state_plus - 1
        if (
            frac_state == 1
        ):  # first state with no prior dose delivered so we dont loop through BEDNT
            future_OAR = accumulated_OAR_dose + OAR_dose[actual_fraction_sf]
            future_tumor = accumulated_tumor_dose + tumor_dose
            future_OAR[
                future_OAR > OAR_limit
            ] = upperbound_normal_tissue  # any dose surpassing the upper bound will be set to the upper bound which will be penalised strongly
            future_tumor[future_tumor > tumor_goal] = upperbound_tumor
            future_values_prob = (Values[index - 1] * prob).sum(
                axis=2
            )  # future values of tumor and oar state
            value_interpolation = RegularGridInterpolator(
                (BEDT, BEDNT), future_values_prob
            )
            future_value_actual = value_interpolation(
                np.array([future_tumor, future_OAR]).T
            )
            Vs = future_value_actual - OAR_dose[actual_fraction_sf]
            actual_policy = Vs.argmax(axis=0)

        elif (
            frac_state == fraction
        ):  # if we are in the actual fraction we do not need to check all possible BED states but only the one we are in
            if fraction != number_of_fractions:
                future_OAR = accumulated_OAR_dose + OAR_dose[actual_fraction_sf]
                overdosing = (future_OAR - OAR_limit).clip(min=0)
                future_OAR[
                    future_OAR > OAR_limit
                ] = upperbound_normal_tissue  # any dose surpassing the upper bound will be set to the upper bound which will be penalised strongly
                future_tumor = accumulated_tumor_dose + tumor_dose
                future_tumor[future_tumor > tumor_goal] = upperbound_tumor
                future_values_prob = (Values[index - 1] * prob).sum(
                    axis=2
                )  # future values of tumor and oar state
                value_interpolation = RegularGridInterpolator(
                    (BEDT, BEDNT), future_values_prob
                )
                penalties = (
                    overdosing * -10000000000
                )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                future_value_actual = value_interpolation(
                    np.array([future_tumor, future_OAR]).T
                )
                Vs = future_value_actual - OAR_dose[actual_fraction_sf] + penalties
                actual_policy = Vs.argmax(axis=0)
            else:
                sf_end = sparing_factors[-1]
                best_action_BED = convert_to_physical(OAR_limit-accumulated_OAR_dose, abn, sf_end)
                best_action_tumor = convert_to_physical(tumor_goal-accumulated_tumor_dose, abt)
                best_action = np.min([best_action_BED, best_action_tumor], axis=0)
                if accumulated_OAR_dose > OAR_limit or accumulated_tumor_dose > tumor_goal:
                    best_action = np.ones(best_action.shape) * min_dose
                if best_action > max_dose:
                    best_action = max_dose
                if best_action < min_dose:
                    best_action = min_dose

                future_tumor = accumulated_tumor_dose + BED_calc0(best_action, abt)
                future_OAR = accumulated_OAR_dose + BED_calc0(best_action, abn, sf_end)
                actual_policy = best_action * 10
        else:
            future_values_prob = (Values[index - 1] * prob).sum(
                axis=2
            )  # future values of tumor and oar state
            value_interpolation = RegularGridInterpolator(
                (BEDT, BEDNT), future_values_prob
            )  # interpolation function
            for tumor_index, tumor_value in enumerate(BEDT):
                for OAR_index, OAR_value in enumerate(
                    BEDNT
                ):  # this and the next for loop allow us to loop through all states
                    future_OAR = OAR_dose + OAR_value
                    overdosing = (future_OAR - OAR_limit).clip(min=0)
                    future_OAR[
                        future_OAR > OAR_limit
                    ] = upperbound_normal_tissue  # any dose surpassing 90.1 is set to 90.1
                    future_tumor = tumor_value + tumor_dose
                    future_tumor[
                        future_tumor > tumor_goal
                    ] = upperbound_tumor  # any dose surpassing the tumor bound is set to tumor_bound + 0.1

                    if (
                        frac_state == number_of_fractions
                    ):  # last state no more further values to add
                        best_action_BED = convert_to_physical(OAR_limit-OAR_value, abn, sf) 
                        # calculate maximal dose that can be delivered to OAR and tumor
                        best_action_tumor = convert_to_physical(tumor_goal-tumor_value, abt, sf*0+1)
                        best_action = np.min(
                            [best_action_BED, best_action_tumor], axis=0
                        )  # take the smaller of both doses to not surpass the limit
                        best_action[best_action > max_dose] = max_dose
                        best_action[best_action < min_dose] = min_dose
                        if (
                            OAR_value > OAR_limit or tumor_value > tumor_goal
                        ):  # if the limit is already surpassed we add a penaltsy
                            best_action = np.ones(best_action.shape) * min_dose
                        future_OAR = OAR_value + BED_calc0(best_action, abn, sf)
                        future_tumor = tumor_value + BED_calc0(best_action, abt, 1)
                        overdose_penalty2 = np.zeros(
                            best_action.shape
                        )  # we need a second penalty if we overdose in the last fraction
                        overdose_penalty3 = np.zeros(best_action.shape)
                        overdose_penalty2[
                            future_tumor > tumor_goal + 0.0001
                        ] = -100000000000
                        overdose_penalty3[
                            future_OAR > OAR_limit + 0.0001
                        ] = (
                            -100000000000
                        )  # A small number has to be added as sometimes 90. > 90 was True
                        end_penalty = (
                            -abs(future_tumor - tumor_goal) * underdosepenalty
                        )  # the farther we are away from the prescribed dose, the higher the penalty. Under- and overdosing is punished
                        end_penalty_OAR = (
                            -(future_OAR - OAR_limit).clip(min=0) * 1000
                        )  # if overdosing the OAR is not preventable, the overdosing should stay as low as possible
                        Values[index][tumor_index][OAR_index] = (
                            end_penalty
                            - BED_calc0(best_action, abn, sf)
                            + overdose_penalty2
                            + overdose_penalty3
                            + end_penalty_OAR
                        )  # we also substract all the dose delivered to the OAR so the algorithm tries to minimise it
                        policy[index][tumor_index][OAR_index] = best_action * 10
                    else:
                        future_value = np.zeros([len(sf), len(actionspace)])
                        for actual_sf in range(0, len(sf)):
                            future_value[actual_sf] = value_interpolation(
                                np.array([future_tumor, future_OAR[actual_sf]]).T
                            )
                        penalties = (
                            overdosing * -10000000000
                        )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                        Vs = future_value - OAR_dose + penalties
                        best_action = Vs.argmax(axis=1)
                        valer = Vs.max(axis=1)
                        policy[index][tumor_index][OAR_index] = best_action
                        Values[index][tumor_index][OAR_index] = valer
    if fraction != number_of_fractions:
        physical_dose = actionspace[actual_policy]
    else:
        physical_dose = actual_policy / 10
    tumor_dose = BED_calc0(physical_dose, abt)
    OAR_dose = BED_calc0(physical_dose, abn, sparing_factors[-1])
    accumulated_tumor_dose = BED_calc0(physical_dose, abt) + accumulated_tumor_dose
    accumulated_OAR_dose = BED_calc0(physical_dose, abn, sparing_factors[-1]) + accumulated_OAR_dose
    return [physical_dose, tumor_dose, OAR_dose]
    # return [
    #     physical_dose,
    #     accumulated_tumor_dose,
    #     accumulated_OAR_dose,
    #     tumor_dose,
    #     OAR_dose,
    # ]