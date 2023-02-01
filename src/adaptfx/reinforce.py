# -*- coding: utf-8 -*-
import numpy as np
import adaptfx as afx
nme = __name__

def min_oar_bed(keys, sets=afx.SETTING_DICT):
    # oar_minimisation is a subset of number of fraction minimisation
    # c has to be zero
    keys.c = 0
    output = min_n_frac(keys, sets=sets)

    return output


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
    prob_update = keys.prob_update
    fixed_mean = keys.fixed_mean
    fixed_std = keys.fixed_std
    shape = keys.shape
    scale = keys.scale
    shape_inv = keys.shape_inv
    scale_inv = keys.scale_inv
    tumor_goal = keys.tumor_goal
    c = keys.c
    abt = keys.abt
    abn = keys.abn
    min_dose = keys.min_dose
    max_dose = keys.max_dose
    # ---------------------------------------------------------------------- #
    # check in which fraction data should be returned for plotting
    policy_plot = 1 if sets.plot_policy == fraction else 0
    values_plot = 1 if sets.plot_values == fraction else 0
    remains_plot = 1 if sets.plot_remains == fraction else 0

    # prepare distribution
    actual_sf = sparing_factors_public[fraction]
    if prob_update == 0:
        # fixed normal distribution for sparing factor
        mean = fixed_mean
        std = fixed_std
        rv = afx.truncated_normal(mean, std, sets.sf_low, sets.sf_high)
    elif prob_update == 1:
        # update normal distribution with gamma prior
        mean = np.mean(sparing_factors_public)
        std = afx.std_posterior(sparing_factors_public, shape, scale)
        rv = afx.truncated_normal(mean, std, sets.sf_low, sets.sf_high)
    elif prob_update == 2:
        # update distribution with inverse-gamma prior
        # posterior predictive distribution is student-t distribution
        rv = afx.student_t(sparing_factors_public, shape_inv, scale_inv)
    else:
        afx.aft_error('invalid "prob_update" key was set', nme)
    # initialise distribution from random variable (rv)
    [sf, prob] = afx.sf_probdist(rv, sets.sf_low, sets.sf_high,
        sets.sf_stepsize, sets.sf_prob_threshold)
    n_sf = len(sf)

    # actionspace
    exp = afx.find_exponent(sets.dose_stepsize)
    accumulated_tumor_dose = np.round(accumulated_tumor_dose, -exp)
    remaining_bed = np.round(tumor_goal - accumulated_tumor_dose, -exp)
    n_bedsteps = int(np.round(remaining_bed / sets.dose_stepsize, 0))
    n_statesteps = int(np.round(remaining_bed / sets.state_stepsize, 0))
    max_physical_dose = afx.convert_to_physical(remaining_bed, abt)

    # automatic max_dose calculation
    if max_dose == -1:
        max_dose = max_physical_dose
    # Reduce max_dose to prohibit tumor_goal overshoot (efficiency)
    max_dose = min(max_dose, max_physical_dose)

    # actionspace in bed dose
    bedt_space = np.linspace(0, remaining_bed, n_bedsteps + 1)
    actionspace = afx.convert_to_physical(bedt_space, abt)
    range_action = (actionspace >= min_dose) & (actionspace <= max_dose)
    actionspace = actionspace[range_action]
    # bed_space to relate actionspace to oar- and tumor-dose
    bedt_space = bedt_space[range_action]
    if not range_action.any():
        # check if actionspace is empty
        bedt_space = np.array([min_dose])
        actionspace = afx.convert_to_physical(bedt_space, abt)
    bedn_space = afx.bed_calc0(actionspace, abn, actual_sf)
    n_action = len(actionspace)

    # tumor bed states for tracking dose
    tumor_limit = tumor_goal + sets.state_stepsize
    # include at least one more step for bedt
    # define number of bed_dose steps to fulfill stepsize
    bedt_states = np.linspace(accumulated_tumor_dose,
        tumor_goal, n_statesteps + 1)
    remaining_states = bedt_states - accumulated_tumor_dose
    n_bedt_states = len(bedt_states)
    
    # relate actionspace to bed and possible sparing factors
    # necessary reshape for broadcasting in value calculation
    bedn_sf_space = afx.bed_calc_matrix(actionspace, abn, sf).reshape(1, n_action, n_sf)
    # values matrix
    # dim(values) = dim(policy) = fractions_remaining * bedt * sf
    n_remaining_fractions = number_of_fractions - fraction
    values = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
    if policy_plot or values_plot or remains_plot:
        policy = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
        remains = np.zeros((n_remaining_fractions + 1, n_bedt_states, n_sf))
    
    finished = False
    # ------------------------------------------------------------------------------------- #
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
            c_penalties = np.where(np.round(future_bedt, -exp) < tumor_goal, -c, 0)
            # for discrete matching
            # future_values = future_values_discrete[np.where(np.isclose(bedt_states, future_bedt))]
            future_values = afx.interpolate(future_bedt, bedt_states, future_values_discrete)
            vs = -bedn_space + future_values + c_penalties
            # argmax of vs along axis 0 to find best action fot the actual sf
            action_index = vs.argmax(axis=0)

            if policy_plot or values_plot or remains_plot:
                # for the policy plot
                future_bedt_full = bedt_states.reshape(n_bedt_states, 1) + bedt_space
                future_bedt_full = np.where(future_bedt_full > tumor_goal, tumor_limit, future_bedt_full)
                c_penalties_full = np.where(future_bedt_full < tumor_goal, -c, 0).reshape(n_bedt_states, n_action, 1)
                future_values_full = afx.interpolate(future_bedt_full, bedt_states, future_values_discrete)
                vs_full = -bedn_sf_space + future_values_full.reshape(n_bedt_states, n_action, 1) + c_penalties_full
                # check vs along the sf axis
                current_policy = bedt_space[vs_full.argmax(axis=1)]
                # ensure that for the goal reached the value/policy is zero (min_dose)
                current_policy[bedt_states == tumor_goal] = 0
                future_remains_discrete = (remains[fraction_index + 1] * prob).sum(axis=1)
                future_bedt_opt = current_policy + bedt_states.reshape(n_bedt_states, 1)
                future_remains = afx.interpolate(future_bedt_opt, bedt_states, future_remains_discrete)
                current_remains = np.where((current_policy - remaining_states[::-1].reshape(n_bedt_states, 1)) >= 0, 0, 1)
                # write to arrays
                policy[fraction_index] = current_policy
                values[fraction_index] = vs_full.max(axis=1)
                remains[fraction_index] = current_remains + future_remains

        elif fraction == number_of_fractions:
            # in the last fraction value is not relevant
            # max_dose is the already calculated remaining dose
            # this should compensate the discretisation of the state space
            action_index = np.abs(remaining_bed - bedt_space).argmin()
            
            if policy_plot:
                policy[fraction_index][0] = bedt_space[action_index]

        elif fraction_state == number_of_fractions:
            # final state to initialise terminal reward
            # dose remaining to be delivered, this is the actionspace in bedt
            last_bed_actions = np.round(tumor_goal - bedt_states, -exp)
            last_actions = afx.convert_to_physical(last_bed_actions, abt)
            # cut the actionspace to min and max dose constraints
            last_bed_actions = np.where(last_actions > max_dose, bedt_space[-1], last_bed_actions)
            last_bed_actions = np.where(last_actions < min_dose, bedt_space[0], last_bed_actions)
            last_actions = np.where(last_actions > max_dose, actionspace[-1], last_actions)
            last_actions = np.where(last_actions < min_dose, actionspace[0], last_actions)
            last_bedn = afx.bed_calc_matrix(last_actions, abn, sf)
            # this smooths out the penalties in underdose and overdose regions
            bedt_diff = np.round(bedt_states + last_bed_actions - tumor_goal, -exp)
            penalties = np.where(bedt_diff == 0, 0, -np.abs(bedt_diff) * sets.inf_penalty)
            # to each best action add the according penalties
            # penalties need to be reshaped for broadcasting
            vs = -last_bedn + penalties.reshape(n_bedt_states, 1)
            values[fraction_index] = vs
            # ensure that for the goal reached the value/poliy is zero (min_dose)
            values[fraction_index][bedt_states==tumor_goal] = 0

            if policy_plot:
                # policy calculation for each bedt, but sf is not considered
                policy[fraction_index] += last_bed_actions.reshape(n_bedt_states, 1)
                # ensure that for the goal reached the value/poliy is zero (min_dose)
                policy[fraction_index][bedt_states==tumor_goal] = 0

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
            # every row of values_penalties is transposed and copied n_sf times
            vs = -bedn_sf_space + future_values.reshape(n_bedt_states, n_action, 1) + c_penalties
            # check vs along the sf axis
            values[fraction_index] = vs.max(axis=1)
            # ensure that for the goal reached the value/poliy is zero (min_dose)
            values[fraction_index][bedt_states==tumor_goal] = 0

            if policy_plot or remains_plot:
                current_policy = bedt_space[vs.argmax(axis=1)]
                # ensure that for the goal reached the value/policy is zero (min_dose)
                current_policy[bedt_states == tumor_goal] = 0
                future_remains_discrete = (remains[fraction_index + 1] * prob).sum(axis=1)
                future_bedt_opt = current_policy + bedt_states.reshape(n_bedt_states, 1)
                future_remains = afx.interpolate(future_bedt_opt, bedt_states, future_remains_discrete)
                current_remains = np.where((current_policy - remaining_states[::-1].reshape(n_bedt_states, 1)) >= 0, 0, 1)
                # write to arrays
                policy[fraction_index] = current_policy
                remains[fraction_index] = current_remains + future_remains

    output = afx.DotDict({})

    output.physical_dose = actionspace[action_index] if not finished else np.nan
    output.tumor_dose = bedt_space[action_index] if not finished else np.nan
    output.oar_dose = bedn_space[action_index] if not finished else np.nan

    if policy_plot:
        output.policy = {}
        output.policy.val = policy
        output.policy.sf = sf
        output.policy.states = remaining_states
    if values_plot:
        output.value = {}
        output.value.val = values
        output.value.sf = sf
        output.value.states = remaining_states
    if remains_plot:
        output.remains = {}
        output.remains.val = remains
        output.remains.sf = sf
        output.remains.states = remaining_states

    return output