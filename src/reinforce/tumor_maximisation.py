import numpy as np
import common.constants as C
from scipy.interpolate import interp2d
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
    accumulated_oar_dose,
    sparing_factors,
    alpha,
    beta,
    oar_limit,
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
    max_physical_dose = convert_to_physical(oar_limit, abn, actual_sf)
    if max_dose == -1:
        max_dose = max_physical_dose
    elif max_dose > max_physical_dose:
        # Reduce max_dose to prohibit oar_limit overshoot
        # in order to check fewer actions for efficiency
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - dose_stepsize
    actionspace = np.arange(min_dose, max_dose + dose_stepsize, dose_stepsize)

    # oar bed for tracking dose
    bedn_limit = oar_limit + dose_stepsize
    bedn = np.arange(accumulated_oar_dose, bedn_limit, dose_stepsize)
    n_bedn = len(bedn)

    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedn * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions, n_bedn, n_sf))
    policy = np.zeros((n_remaining_fractions, n_bedn, n_sf))

    # tumor bed reward
    # dim(bedt_reward) = sf * actionspace
    delivered_oar_dose = BED_calc_matrix(sf, abn, actionspace)
    bedt_reward_action = BED_calc_matrix(1, abt, actionspace)
    bedt_reward, _ = np.meshgrid(bedt_reward_action, np.zeros(n_sf))
    # ---------------------------------------------------------------------- #
    remaining_fractions = np.arange(number_of_fractions, fraction - 1, -1)
    # for fraction_index, fraction_state in enumerate(remaining_fractions):
    #     if fraction_state == 1:
    #         # first state with no prior dose delivered
    #         # so we dont loop through BEDT
    #         future_bedn = accumulated_oar_dose + delivered_oar_dose
    #         future_bedn[future_bedn > oar_limit] = bedn_limit
    #         value = interp2d(sf, bedn)
    #     elif fraction_state == fraction and fraction != number_of_fractions:
    #         # state is the actual fraction to calculate
    #         # but actual fraction is not the last fraction

    #     elif fraction == number_of_fractions:
    #         # in the last fraction value is not relevant
    #     else:
    #         for bedn_index, bedn_value in enumerate(bedn):


    return [0,0,0]
        