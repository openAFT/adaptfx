# -*- coding: utf-8 -*-
import numpy as np
import constants as C
from scipy.interpolate import interp1d
from maths import (std_calc, 
                   truncated_normal,
                   sf_probdist)
from radiobiology import (bed_calc_matrix,
                          bed_calc0,
                          convert_to_physical)

def value_eval(
    fraction,
    number_of_fractions,
    accumulated_tumor_dose,
    sparing_factors,
    alpha,
    beta,
    tumor_goal,
    abt,
    abn,
    min_dose,
    max_dose,
    fixed_prob,
    fixed_mean,
    fixed_std,
    sf_low=C.SF_LOW,
    sf_high=C.SF_HIGH,
    sf_stepsize=C.SF_STEPSIZE,
    sf_prob_threshold=C.SF_PROB_THRESHOLD,
    inf_penalty=C.INF_PENALTY,
    dose_stepsize=C.DOSE_STEP_SIZE
):
    # ---------------------------------------------------------------------- #
    underdosage = 1 # sort this out!!!!
    dose_stepsize = 5
    sf_stepsize = 0.3
    dose_margin = 1e-4
    # dose_resolution
    # np.ones(n_sf)
    # dose_step size actionspace

    # prepare distribution
    actual_sf = sparing_factors[-1]
    if not fixed_prob:
        # setup the sparingfactor distribution
        mean = np.mean(sparing_factors)
        std = std_calc(sparing_factors, alpha, beta)
    else:
        mean = fixed_mean
        std = fixed_std
    # initialise normal distributed random variable
    X = truncated_normal(mean, std, sf_low, sf_high)
    sf, prob = sf_probdist(X, sf_low, sf_high, sf_stepsize, sf_prob_threshold)
    n_sf = len(sf)

    # actionspace
    max_physical_dose = convert_to_physical(tumor_goal, abt)
    if max_dose == -1:
        # automatic max_dose calculation
        max_dose = max_physical_dose
    elif max_dose > max_physical_dose:
        # Reduce max_dose to prohibit tumor_goal overshoot
        # in order to check fewer actions for efficiency
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - dose_stepsize
    actionspace = np.arange(min_dose, max_dose + dose_stepsize/10, dose_stepsize/10)
    n_action = len(actionspace)

    # tumor bed for tracking dose
    tumor_limit = tumor_goal + dose_stepsize
    bedt = np.arange(accumulated_tumor_dose, tumor_limit, dose_stepsize)
    n_bedt = len(bedt)

    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedt * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions, n_bedt, n_sf))
    policy = np.zeros((n_remaining_fractions, n_bedt, n_sf))
    # ---------------------------------------------------------------------- #
    remaining_fractions = np.arange(number_of_fractions, fraction - 1, -1)
    for fraction_index, fraction_state in enumerate(remaining_fractions):
        if fraction_state == 1:
            # first state with no prior dose delivered
            # so we dont loop through BEDT
            bedn_space = bed_calc_matrix(actionspace, abn, actual_sf)
            bedt_space = bed_calc0(actionspace, abt)
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt, future_values_discrete)
            print(bedt)
            print(bedt_space)
            future_values = future_values_func(bedt_space)
            vs = -bedn_space + future_values

        elif fraction_state == fraction and fraction != number_of_fractions:
            # state is the actual fraction to calculate
            # but actual fraction is not the last fraction
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            bedn_space = bed_calc_matrix(actionspace, abn, actual_sf)
            bedt_space = bed_calc_matrix(actionspace, abt, np.ones(n_sf))
            future_bedt = accumulated_tumor_dose + bedt_space
            overdose_args = np.where(future_bedt > tumor_goal)[0]
            future_bedt[overdose_args] = tumor_goal + dose_margin
            penalties = np.zeros((n_action, n_sf))
            penalties[overdose_args] = -inf_penalty
            future_values_func = interp1d(bedt, future_values_discrete)
            future_values = future_values_func(future_bedt)
            vs = -bedn_space + future_values + penalties

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant
            best_actions = convert_to_physical(tumor_goal-accumulated_tumor_dose, abt)
            if best_actions < min_dose:
                best_actions = min_dose
            if best_actions > max_dose:
                best_actions = max_dose
            last_bedn = bed_calc0(best_actions, abn, actual_sf)

        elif fraction_index == 0:
            # final state to initialise terminal reward
            best_actions = convert_to_physical(tumor_goal - bedt, abt)
            best_actions[best_actions < min_dose] = min_dose
            best_actions[best_actions > max_dose] = max_dose
            last_bedn = bed_calc_matrix(best_actions, abn, sf)
            last_bedt = bedt + bed_calc0(best_actions, abt, 1)
            penalties = (last_bedt - tumor_goal) * underdosage
            penalties[np.abs(penalties) < dose_margin] = 0
            penalties[penalties > 0] = -inf_penalty
            # to each best action add the according penalties
            vs = -last_bedn + penalties.reshape(n_bedt, 1)

            values[fraction_index] = vs
            # policy calculation for each bedt, but sf is not considered
            _, police = np.meshgrid(np.ones(n_sf), best_actions)
            policy[fraction_index] = police

        elif fraction_index != 0:
            # every other state but the last
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt, future_values_discrete)
            bedn_space = bed_calc_matrix(actionspace, abn, sf)
            bedt_space = bed_calc_matrix(actionspace, abt, np.ones(n_sf))
            for bedt_index, bedt_dose in enumerate(bedt): 
                future_bedt = bedt_space + bedt_dose
                future_bedt[future_bedt > tumor_goal] = tumor_goal + dose_margin
                future_values = future_values_func(future_bedt)
                penalties = np.zeros((n_action, n_sf))
                penalties[future_bedt > tumor_goal] = -inf_penalty
                # to each action and sparing factor add future values and penalties
                vs = -bedn_space + future_values + penalties

                values[fraction_index][bedt_index] = vs.max(axis=0)
                policy[fraction_index][bedt_index] = vs.argmax(axis=0)
        
        else:
            print('error')
                


    return [0,0,0]
        