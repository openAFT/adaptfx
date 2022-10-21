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

def min_oar_bed(
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
    # prepare distribution
    actual_sf = sparing_factors[-1]
    if not fixed_prob:
        # setup the sparingfactor distribution
        mean = np.mean(sparing_factors)
        std = std_calc(sparing_factors, alpha, beta)
    else:
        mean = fixed_mean
        std = fixed_std
    # initialise normal distributed random variable (rv)
    rv = truncated_normal(mean, std, sf_low, sf_high)
    sf, prob = sf_probdist(rv, sf_low, sf_high, sf_stepsize, sf_prob_threshold)
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

    # tumor bed for tracking dose
    remaining_dose = tumor_goal - accumulated_tumor_dose
    # include at least one more step for bedt
    bed_diff = remaining_dose + dose_stepsize
    # define number of bed_dose steps to fulfill stepsize
    # this line just rounds up the number of steps
    n_bedsteps = int(bed_diff // dose_stepsize + (bed_diff % dose_stepsize > 0))
    tumor_limit = tumor_goal + dose_stepsize
    bedt = np.linspace(accumulated_tumor_dose, tumor_limit, n_bedsteps)
    n_bedt = len(bedt)

    # actionspace in physical dose
    actions_bedt = np.linspace(0, remaining_dose, n_bedsteps)
    actions_physical = convert_to_physical(actions_bedt, abt)
    range_condition = (actions_physical >= min_dose) & (actions_physical <= max_dose)
    actionspace = actions_physical[range_condition]
    n_action = len(actionspace)

    # bed_space to relate actionspace to oar- and tumor-dose
    bedn_space = bed_calc0(actionspace, abn, actual_sf)
    bedt_space = bed_calc0(actionspace, abt)
    # relate actionspace to bed and possible sparing factors
    bedn_sf_space = bed_calc_matrix(actionspace, abn, sf)
    _ , bedt_sf_space = np.meshgrid(np.ones(n_sf), bedt_space)
    # note the line below is equivalent but 30% slower:
    # -bedt_sf_space = bed_calc_matrix(actionspace, abt, np.ones(n_sf))- #

    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedt * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions, n_bedt, n_sf))
    policy = np.zeros((n_remaining_fractions, n_bedt, n_sf))
    
    #initialise physical dose scalar
    physical_dose = 0
    # ---------------------------------------------------------------------- #
    remaining_fractions = np.arange(number_of_fractions, fraction - 1, -1)
    for fraction_index, fraction_state in enumerate(remaining_fractions):
        if fraction_state == 1:
            # first state with no prior dose delivered
            # so we dont loop through BEDT
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt, future_values_discrete)
            future_values = future_values_func(bedt_space)
            vs = -bedn_space + future_values
            physical_dose = float(actionspace[vs.argmax(axis=0)])
            break

        elif fraction_state == fraction and fraction != number_of_fractions:
            # state is the actual fraction to calculate
            # but actual fraction is not the last fraction
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt, future_values_discrete)
            future_bedt = accumulated_tumor_dose + bedt_space
            future_values = future_values_func(future_bedt)
            overdose_args = (future_bedt > tumor_goal)
            future_bedt[overdose_args] = tumor_limit
            penalties = np.zeros(n_action)
            penalties[overdose_args] = -inf_penalty
            vs = -bedn_space + future_values + penalties
            # print(actionspace)
            # print(vs)
            physical_dose = float(actionspace[vs.argmax(axis=0)])
            break

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant
            best_actions = convert_to_physical(remaining_dose, abt)
            if best_actions < min_dose:
                best_actions = min_dose
            if best_actions > max_dose:
                best_actions = max_dose
            physical_dose = best_actions
            break

        elif fraction_index == 0:
            # final state to initialise terminal reward
            # dose remaining to be delivered, this is the actionspace in bedt
            remaining_bedt = tumor_goal - bedt
            min_dose_bed = bed_calc0(min_dose, abt)
            max_dose_bed = bed_calc0(max_dose, abt)
            # cut the actionspace to min and max dose constraints
            remaining_bedt[remaining_bedt < min_dose_bed] = min_dose_bed
            remaining_bedt[remaining_bedt > max_dose_bed] = max_dose_bed
            best_actions = convert_to_physical(remaining_bedt, abt)
            last_bedn = bed_calc_matrix(best_actions, abn, sf)
            last_bedt = bedt + remaining_bedt
            penalties = last_bedt - tumor_goal
            penalties[penalties > 0] = -inf_penalty
            # to each best action add the according penalties
            # penalties need to be reshaped as it was not numpy allocated
            vs = -last_bedn + penalties.reshape(n_bedt, 1)

            values[fraction_index] = vs
            # policy calculation for each bedt, but sf is not considered
            _, police = np.meshgrid(np.ones(n_sf), best_actions)
            policy[fraction_index] = police

        elif fraction_index != 0:
            # every other state but the last
            # this calculates the value function in the future fractions
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt, future_values_discrete)
            for bedt_index, bedt_dose in enumerate(bedt):
                future_bedt = bedt_sf_space + bedt_dose
                overdose_args = future_bedt > tumor_goal
                future_bedt[overdose_args] = tumor_limit
                future_values = future_values_func(future_bedt)
                penalties = np.zeros((n_action, n_sf))
                penalties[overdose_args] = -inf_penalty
                # to each action and sparing factor add future values and penalties
                vs = -bedn_sf_space + future_values + penalties

                values[fraction_index][bedt_index] = vs.max(axis=0)
                policy[fraction_index][bedt_index] = vs.argmax(axis=0)
    
    tumor_dose = np.round(bed_calc0(physical_dose, abt), 2)
    oar_dose = np.round(bed_calc0(physical_dose, abn, actual_sf), 2)
    dose = np.round(physical_dose, 2)

    return [dose, tumor_dose, oar_dose]
        