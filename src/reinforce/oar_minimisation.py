import numpy as np
import common.constants as C
from scipy.interpolate import interp1d
from common.maths import (std_calc, 
                          get_truncated_normal,
                          probdist)
from common.radiobiology import (argfind,
                                 BED_calc_matrix,
                                 BED_calc0,
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
    sf_prob=C.SF_PROB,
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
    X = get_truncated_normal(mean, std, 0, sf_high)
    prob = np.array(probdist(X))
    sf_all = np.arange(sf_low, sf_high, sf_stepsize)
    # get rid of all probabilities below given threshold
    sf = sf_all[prob > sf_prob]
    n_sf = len(sf)

    # actionspace
    max_physical_dose = convert_to_physical(tumor_goal, abt)
    if max_dose == -1:
        max_dose = max_physical_dose
    elif max_dose > max_physical_dose:
        # Reduce max_dose to prohibit tumor_goal overshoot
        # in order to check fewer actions for efficiency
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - dose_stepsize
    actionspace = np.arange(min_dose, max_dose + dose_stepsize, dose_stepsize)

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

        elif fraction_state == fraction and fraction != number_of_fractions:
            # state is the actual fraction to calculate
            # but actual fraction is not the last fraction

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant

        elif fraction_index == 0:
            # final state to initialise terminal reward
            best_action = convert_to_physical()
            if best_action < min_dose or best_action > max_dose
                best_action = argfind(actionspace, best_action)
        else: 
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_values_func = interp1d(bedt, future_values_discrete)
            bedn_space = BED_calc_matrix(sf,abn,actionspace)
            bedt_space = BED_calc_matrix(np.ones(n_sf),abt,actionspace)
            for bedt_index, bedt_dose in enumerate(bedt):                
                future_bedt = bedt_space + bedt_dose
                future_bedt[future_bedt > tumor_goal] = tumor_limit
                future_values = future_values_func(future_bedt)
                penalties = np.zeros(n_sf)
                penalties[future_bedt > tumor_goal] = inf_penalty

                vs = -bedn_space + future_values + penalties

                values[fraction_index][bedt_index] = vs.max(axis=1)
                policy[fraction_index][bedt_index] = vs.argmax(axis=1)
                


    return [0,0,0]
        