# -*- coding: utf-8 -*-
import numpy as np
import adaptfx as afx

def min_oar_bed(keys, sets=afx.SETTING_DICT):
    # check if keys is a dictionary from manual user
    if isinstance(keys, dict):
        keys = afx.DotDict(keys)

    if isinstance(sets, dict):
        sets = afx.DotDict(sets)

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
    # prepare distribution
    actual_sf = sparing_factors_public[-1]
    if not fixed_prob:
        # setup the sparingfactor distribution
        mean = np.mean(sparing_factors_public)
        std = afx.std_calc(sparing_factors_public, alpha, beta)
    else:
        mean = fixed_mean
        std = fixed_std
    # initialise normal distributed random variable (rv)
    rv = afx.truncated_normal(mean, std, sets.sf_low, sets.sf_high)
    [sf, prob] = afx.sf_probdist(rv, sets.sf_low, sets.sf_high,
                            sets.sf_stepsize, sets.sf_prob_threshold)
    n_sf = len(sf)

    # actionspace
    max_physical_dose = afx.convert_to_physical(tumor_goal, abt)
    dose_stepsize = afx.convert_to_physical(sets.bedt_stepsize, abt)
    if max_dose == -1:
        # automatic max_dose calculation
        max_dose = max_physical_dose
    elif max_dose > max_physical_dose:
        # Reduce max_dose to prohibit tumor_goal overshoot
        # in order to check fewer actions for efficiency
        max_dose = max_physical_dose
    if min_dose > max_dose:
        min_dose = max_dose - dose_stepsize

    # tumor bed states for tracking dose
    remaining_dose = tumor_goal - accumulated_tumor_dose
    # include at least one more step for bedt
    state_diff = remaining_dose + sets.state_stepsize
    # define number of bed_dose steps to fulfill stepsize
    # this line just rounds up the number of steps
    n_statesteps = int(state_diff // sets.state_stepsize + 
                    (state_diff % sets.state_stepsize > 0))
    state_limit = tumor_goal + sets.state_stepsize
    bedt_states = np.linspace(accumulated_tumor_dose, state_limit, n_statesteps + 1)
    n_bedt_states = len(bedt_states)

    # actionspace in physical dose
    bed_diff = remaining_dose + sets.bedt_stepsize
    n_bedsteps = int(bed_diff // sets.bedt_stepsize + 
                    (bed_diff % sets.bedt_stepsize > 0))
    tumor_limit = tumor_goal + sets.bedt_stepsize
    actions_bedt = np.linspace(0, remaining_dose, n_bedsteps)
    actions_physical = afx.convert_to_physical(actions_bedt, abt)
    range_condition = (actions_physical >= min_dose) & \
                        (actions_physical <= max_dose)
    actionspace = actions_physical[range_condition]
    n_action = len(actionspace)

    # bed_space to relate actionspace to oar- and tumor-dose
    bedn_space = afx.bed_calc0(actionspace, abn, actual_sf)
    bedt_space = afx.bed_calc0(actionspace, abt)
    # relate actionspace to bed and possible sparing factors
    # necessary reshape for broadcasting in value calculation
    bedn_sf_space = afx.bed_calc_matrix(actionspace, abn, sf).reshape(1, n_action, n_sf)

    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedt * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions, n_bedt_states, n_sf))
    # policy = np.zeros((n_remaining_fractions, n_bedt_states, n_sf))
    
    #initialise physical dose scalar
    physical_dose = 0
    # ---------------------------------------------------------------------- #
    remaining_fractions = np.arange(number_of_fractions, fraction - 1, -1)
    for fraction_index, fraction_state in enumerate(remaining_fractions):
        if fraction_state == fraction and fraction != number_of_fractions:
            # state is the actual fraction to calculate
            # e.g. in the first fraction_state there is no prior dose delivered
            # an future_bedt is equal to bedt_space
            # but actual fraction is not the last fraction
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            future_bedt = accumulated_tumor_dose + bedt_space
            overdose_args = (future_bedt > tumor_goal)
            future_bedt = np.where(overdose_args, tumor_limit, future_bedt)
            penalties = np.where(overdose_args, -sets.inf_penalty, 0)
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            vs = -bedn_space + future_values + penalties
            physical_dose = float(actionspace[vs.argmax(axis=0)])

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant
            best_actions = afx.convert_to_physical(remaining_dose, abt)
            if best_actions < min_dose:
                best_actions = min_dose
            if best_actions > max_dose:
                best_actions = max_dose
            physical_dose = best_actions

        elif fraction_state == number_of_fractions:
            # final state to initialise terminal reward
            # dose remaining to be delivered, this is the actionspace in bedt
            remaining_bedt = tumor_goal - bedt_states
            min_dose_bed = afx.bed_calc0(min_dose, abt)
            max_dose_bed = afx.bed_calc0(max_dose, abt)
            # cut the actionspace to min and max dose constraints
            remaining_bedt[remaining_bedt < min_dose_bed] = min_dose_bed
            remaining_bedt[remaining_bedt > max_dose_bed] = max_dose_bed
            best_actions = afx.convert_to_physical(remaining_bedt, abt)
            last_bedn = afx.bed_calc_matrix(best_actions, abn, sf)
            bedt_diff = bedt_states + remaining_bedt - tumor_goal
            penalties = np.where(bedt_diff > 0, -sets.inf_penalty, bedt_diff)
            # to each best action add the according penalties
            # penalties need to be reshaped as it was not numpy allocated
            vs = -last_bedn + penalties.reshape(n_bedt_states, 1)

            values[fraction_index] = vs
            # policy calculation for each bedt, but sf is not considered
            # _, police = np.meshgrid(np.ones(n_sf), best_actions)
            # policy[fraction_index] = police

        elif fraction_index != 0:
            # every other state but the last
            # this calculates the value function in the future fractions
            future_values_discrete = (values[fraction_index - 1] * prob).sum(axis=1)
            # bedt_states is reshaped such that numpy broadcast leads to 2D array
            future_bedt = bedt_states.reshape(n_bedt_states, 1) + bedt_space
            overdose_args = future_bedt > tumor_goal
            future_bedt = np.where(overdose_args, tumor_limit, future_bedt)
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            values_penalties = np.where(overdose_args, future_values-sets.inf_penalty, future_values)
            # dim(bedn_sf_space)=(1,n_action, n_sf), dim(values_penalties)=(n_states, n_action)
            # every row of values_penalties is transposed and copied n_sf times
            vs = -bedn_sf_space + values_penalties.reshape(n_bedt_states, n_action, 1)
            
            values[fraction_index] = vs.max(axis=1)
            # policy[fraction_index] = vs.argmax(axis=1)
    
    tumor_dose = afx.bed_calc0(physical_dose, abt)
    oar_dose = afx.bed_calc0(physical_dose, abn, actual_sf)

    return [physical_dose, tumor_dose, oar_dose]