# -*- coding: utf-8 -*-
import numpy as np
import adaptfx as afx
nme = __name__

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
    # check in which fraction policy should be returned
    policy_plot = 1 if sets.plot_policy == fraction else 0
    
    # prepare distribution
    actual_sf = sparing_factors_public[fraction]
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
    remaining_bed = tumor_goal - accumulated_tumor_dose
    max_physical_dose = afx.convert_to_physical(remaining_bed, abt)

    if max_dose == -1:
        # automatic max_dose calculation
        max_dose = max_physical_dose
    # Reduce max_dose to prohibit tumor_goal overshoot (efficiency)
    max_dose = min(max_dose, max_physical_dose)

    # actionspace in physical dose
    diff_action = afx.step_round(max_dose-min_dose, sets.dose_stepsize)
    physical_action = np.arange(min_dose, diff_action + min_dose, sets.dose_stepsize)
    # step_round rounds down so we include the maxdose
    actionspace = np.append(physical_action, max_dose)
    n_action = len(actionspace)

    # tumor bed states for tracking dose
    tumor_limit = tumor_goal + sets.state_stepsize
    # include at least one more step for bedt
    # define number of bed_dose steps to fulfill stepsize
    bedt_states = np.arange(accumulated_tumor_dose,
        tumor_limit, sets.state_stepsize)
    n_bedt_states = len(bedt_states)

    # bed_space to relate actionspace to oar- and tumor-dose
    bedn_space = afx.bed_calc0(actionspace, abn, actual_sf)
    bedt_space = afx.bed_calc0(actionspace, abt)
    # relate actionspace to bed and possible sparing factors
    # necessary reshape for broadcasting in value calculation
    bedn_sf_space = afx.bed_calc_matrix(actionspace, abn, sf).reshape(1, n_action, n_sf)

    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedt * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
    if policy_plot:
        policy = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
    
    finished = False
    # ---------------------------------------------------------------------- #
    remaining_fractions = np.arange(number_of_fractions, fraction - 1, -1)
    remaining_index = remaining_fractions - fraction
    # note that lowest fraction_state is one not zero
    # and remaining_index counts in python indices
    for fraction_index, fraction_state in zip(remaining_index, remaining_fractions):
        if remaining_bed <= 0:
            finished = True
            break

        elif fraction_state == fraction and fraction != number_of_fractions:
            # state is the actual fraction to calculate
            # e.g. in the first fraction_state there is no prior dose delivered
            # and future_bedt is equal to bedt_space
            future_values_discrete = (values[fraction_index + 1] * prob).sum(axis=1)
            future_bedt = accumulated_tumor_dose + bedt_space
            future_bedt = np.where(future_bedt > tumor_goal, tumor_limit, future_bedt)
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            vs = -bedn_space + future_values
            # argmax of vs along axis 0 to find best action fot the actual sf
            action_index = vs.argmax(axis=0)

            if policy_plot:
                # for the policy plot
                vs_full = (-bedn_sf_space+ future_values.reshape(1, n_action, 1))[0]
                # check vs along the sf axis
                values[fraction_index][0] = vs_full.max(axis=0)
                policy[fraction_index][0] = actionspace[vs_full.argmax(axis=0)]

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant
            # max_dose is the already calculated remaining dose
            # this should compensate the discretisation of the state space
            action_index = np.where(actionspace == max(min_dose, max_dose))

        elif fraction_state == number_of_fractions:
            # final state to initialise terminal reward
            # dose remaining to be delivered, this is the actionspace in bedt
            last_actions = tumor_goal - bedt_states
            min_dose_bed = afx.bed_calc0(min_dose, abt)
            max_dose_bed = afx.bed_calc0(max_dose, abt)
            # cut the actionspace to min and max dose constraints
            last_actions[last_actions < min_dose_bed] = min_dose_bed
            last_actions[last_actions > max_dose_bed] = max_dose_bed
            last_physical_actions = afx.convert_to_physical(last_actions, abt)
            last_bedn = afx.bed_calc_matrix(last_physical_actions, abn, sf)
            # this smooths out the penalties in underdose and overdose regions
            bedt_diff = (bedt_states + last_actions - tumor_goal) * sets.inf_penalty
            penalties = np.where(bedt_diff > 0, -bedt_diff, bedt_diff)
            # to each best action add the according penalties
            # penalties need to be reshaped for broadcasting
            vs = -last_bedn + penalties.reshape(n_bedt_states, 1)
            values[fraction_index] = vs

            if policy_plot:
                # policy calculation for each bedt, but sf is not considered
                policy[fraction_index] += last_physical_actions.reshape(n_bedt_states, 1)

        elif fraction_state != number_of_fractions:
            # every other state but the last
            # this calculates the value function in the future fractions
            future_values_discrete = (values[fraction_index + 1] * prob).sum(axis=1)
            # bedt_states is reshaped such that numpy broadcast leads to 2D array
            future_bedt = bedt_states.reshape(n_bedt_states, 1) + bedt_space
            future_bedt = np.where(future_bedt > tumor_goal, tumor_limit, future_bedt)
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            # dim(bedn_sf_space)=(1,n_action,n_sf),dim(future_values)=(n_states,n_action)
            # every row of values_penalties is transposed and copied n_sf times
            vs = -bedn_sf_space + future_values.reshape(n_bedt_states, n_action, 1)
            # check vs along the sf axis
            values[fraction_index] = vs.max(axis=1)

            if policy_plot:
                policy[fraction_index] = actionspace[vs.argmax(axis=1)]

    if finished:
        output = {'physical_dose': np.nan, 'tumor_dose': np.nan, 
            'oar_dose': np.nan, 'sf': sf, 'states': bedt_states}
    else:
        output = {'physical_dose': actionspace[action_index], 'tumor_dose': bedt_space[action_index], 
            'oar_dose': bedn_space[action_index], 'sf': sf, 'states': bedt_states}
    if policy_plot:
        output['policy'] = policy
    return afx.DotDict(output)

def min_n_frac(keys, sets=afx.SETTING_DICT):
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
    
    # prepare distribution
    actual_sf = sparing_factors_public[fraction]
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
    remaining_bed = tumor_goal - accumulated_tumor_dose
    max_physical_dose = afx.convert_to_physical(remaining_bed, abt)

    if max_dose == -1:
        # automatic max_dose calculation
        max_dose = max_physical_dose
    # Reduce max_dose to prohibit tumor_goal overshoot (efficiency)
    max_dose = min(max_dose, max_physical_dose)
    min_dose = max(0, min(min_dose, max_dose - sets.dose_stepsize))

    # actionspace in physical dose
    diff_action = afx.step_round(max_dose-min_dose, sets.dose_stepsize)
    pre_actionspace = np.arange(min_dose, diff_action + min_dose, sets.dose_stepsize)
    # step_round rounds down so we include the maxdose
    actionspace = np.append(pre_actionspace, max_dose)
    n_action = len(actionspace)

    # tumor bed states for tracking dose
    tumor_limit = tumor_goal + sets.state_stepsize
    # include at least one more step for bedt
    # define number of bed_dose steps to fulfill stepsize
    bedt_states = np.arange(accumulated_tumor_dose,
        tumor_limit, sets.state_stepsize)
    n_bedt_states = len(bedt_states)

    # bed_space to relate actionspace to oar- and tumor-dose
    bedn_space = afx.bed_calc0(actionspace, abn, actual_sf)
    bedt_space = afx.bed_calc0(actionspace, abt)
    # relate actionspace to bed and possible sparing factors
    # necessary reshape for broadcasting in value calculation
    bedn_sf_space = afx.bed_calc_matrix(actionspace, abn, sf).reshape(1, n_action, n_sf)

    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedt * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
    if policy_plot:
        policy = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
    
    # initialise physical dose scalar (the optimal action)
    optimal_action = 0
    # ---------------------------------------------------------------------- #
    remaining_fractions = np.arange(number_of_fractions, fraction - 1, -1)
    remaining_index = remaining_fractions - fraction
    # note that lowest fraction_state is one
    # but the lowest fraction_index is zero
    for fraction_index, fraction_state in zip(remaining_index, remaining_fractions):
        if remaining_bed <= 0:
            optimal_action = 0
            break
        
        elif fraction_state == fraction and fraction != number_of_fractions:
            # state is the actual fraction to calculate
            # e.g. in the first fraction_state there is no prior dose delivered
            # and future_bedt is equal to bedt_space
            future_values_discrete = (values[fraction_index + 1] * prob).sum(axis=1)
            future_bedt = accumulated_tumor_dose + bedt_space
            future_bedt = np.where(future_bedt > tumor_goal, tumor_limit, future_bedt)
            c_penalties = np.where(future_bedt < tumor_goal, -c, 0)
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            vs = -bedn_space + future_values + c_penalties
            # argmax of vs along axis 0 to find best action fot the actual sf
            optimal_action = float(actionspace[vs.argmax(axis=0)])

            if policy_plot:
                # for the policy plot
                vs_full = (-bedn_sf_space+ future_values.reshape(1, n_action, 1))[0]
                # check vs along the sf axis
                values[fraction_index][0] = vs_full.max(axis=0)
                policy[fraction_index][0] = actionspace[vs_full.argmax(axis=0)]

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant
            # max_dose is the already calculated remaining dose
            # this should compensate the discretisation of the state space
            optimal_action = max(min_dose, max_dose)

        elif fraction_state == number_of_fractions:
            # final state to initialise terminal reward
            # dose remaining to be delivered, this is the actionspace in bedt
            last_actions = tumor_goal - bedt_states
            min_dose_bed = afx.bed_calc0(min_dose, abt)
            max_dose_bed = afx.bed_calc0(max_dose, abt)
            # cut the actionspace to min and max dose constraints
            last_actions[last_actions < min_dose_bed] = min_dose_bed
            last_actions[last_actions > max_dose_bed] = max_dose_bed
            optimal_action = afx.convert_to_physical(last_actions, abt)
            last_bedn = afx.bed_calc_matrix(optimal_action, abn, sf)
            # this smooths out the penalties in underdose and overdose regions
            bedt_diff = (bedt_states + last_actions - tumor_goal) * sets.inf_penalty
            penalties = np.where(bedt_diff > 0, -bedt_diff, bedt_diff)
            # to each best action add the according penalties
            # penalties need to be reshaped for broadcasting
            vs = -last_bedn + penalties.reshape(n_bedt_states, 1)
            values[fraction_index] = vs

            if policy_plot:
                # policy calculation for each bedt, but sf is not considered
                policy[fraction_index] += optimal_action.reshape(n_bedt_states, 1)

        elif fraction_state != number_of_fractions:
            # every other state but the last
            # this calculates the value function in the future fractions
            future_values_discrete = (values[fraction_index + 1] * prob).sum(axis=1)
            # bedt_states is reshaped such that numpy broadcast leads to 2D array
            future_bedt = bedt_states.reshape(n_bedt_states, 1) + bedt_space
            future_bedt = np.where(future_bedt > tumor_goal, tumor_limit, future_bedt)
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            c_penalties = np.where(future_bedt < tumor_goal, -c, 0).reshape(n_bedt_states, n_action, 1)
            # dim(bedn_sf_space)=(1,n_action,n_sf),dim(future_values)=(n_states,n_action)
            # every row of future_values is transposed and copied n_sf times
            vs = -bedn_sf_space + future_values.reshape(n_bedt_states, n_action, 1) + c_penalties
            # check vs along the sf axis
            values[fraction_index] = vs.max(axis=1)

            if policy_plot:
                policy[fraction_index] = actionspace[vs.argmax(axis=1)]
    
    tumor_dose = afx.bed_calc0(optimal_action, abt)
    oar_dose = afx.bed_calc0(optimal_action, abn, actual_sf)

    output = {'physical_dose': optimal_action, 'tumor_dose': tumor_dose, 
        'oar_dose': oar_dose, 'sf': sf, 'states': bedt_states}
    if policy_plot:
        output['policy'] = policy
    return afx.DotDict(output)