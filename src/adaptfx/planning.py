# -*- coding: utf-8 -*-
"""
This function computes the doses for a whole
treatment plan retrospectively 
(when all sparing factors are known)

or for a single fraction with known previous
sparing factors and accumulated BED
"""

import numpy as np
import adaptfx as afx
nme = __name__

def multiple(algorithm, keys, sets=afx.SETTING_DICT):
    if isinstance(keys, dict):
        # check if keys is a dictionary from manual user
        keys = afx.DotDict(keys)

    if isinstance(sets, dict):
        # check if keys is a dictionary from manual user
        sets = afx.DotDict(sets)

    if keys.fraction != 0:
        # if only a specific fraction should be calculated
        fractions_list = np.array([keys.fraction])
        physical_doses = np.zeros(1)
        tumor_doses = np.zeros(1)
        oar_doses = np.zeros(1)
    else:
        # for calculation whole treatment in retrospect
        fractions_list = np.arange(1, keys.number_of_fractions + 1, 1)
        physical_doses = np.zeros(keys.number_of_fractions)
        tumor_doses = np.zeros(keys.number_of_fractions)
        oar_doses = np.zeros(keys.number_of_fractions)

    first_tumor_dose = keys.accumulated_tumor_dose
    first_oar_dose = keys.accumulated_oar_dose

    output_whole = afx.DotDict({})

    for i, keys.fraction in enumerate(fractions_list):
        keys.sparing_factors_public = keys.sparing_factors[0 : keys.fraction + 1]
        if algorithm == 'oar':
            output = afx.min_oar_bed(keys, sets)
        elif algorithm == 'frac':
            output = afx.min_n_frac(keys, sets)
            
        elif algorithm == 'oar_old':
            output = afx.min_oar_bed_old(keys, sets)
        elif algorithm == 'frac_old':
            output = afx.min_n_frac_old(keys, sets)
        elif algorithm == 'tumor_old':
            output = afx.max_tumor_bed_old(keys, sets)
        elif algorithm == 'tumor_oar_old':
            output = afx.min_oar_max_tumor_old(keys, sets)
        else:
            afx.aft_error('no valid algorithm given', nme)

        physical_doses[i] = output.physical_dose
        tumor_doses[i] = output.tumor_dose
        oar_doses[i] = output.oar_dose

        keys.accumulated_tumor_dose = np.nansum(tumor_doses) + first_tumor_dose
        keys.accumulated_oar_dose = np.nansum(oar_doses) + first_oar_dose

        # user specifies to plot policy number, if equal to fraction plot
        # if both zero than the user doesn't want to plot policy
        if sets.plot_policy == keys.fraction:
            output_whole.policy = output.policy
            output_whole.policy.fractions = fractions_list[sets.plot_policy - 1:]
        if sets.plot_values == keys.fraction:
            output_whole.value = output.value
            output_whole.value.fractions = fractions_list[sets.plot_values - 1:]
        if sets.plot_remains == keys.fraction:
            output_whole.remains = output.remains
            output_whole.remains.fractions = fractions_list[sets.plot_remains - 1:]

    # store doses
    exponent = afx.find_exponent(sets.dose_stepsize) - 1
    [output_whole.physical_doses, output_whole.tumor_doses, output_whole.oar_doses] = np.around(
        [physical_doses, tumor_doses, oar_doses], -exponent)
    output_whole.oar_sum, output_whole.tumor_sum = np.around(
        [np.nansum(oar_doses), np.nansum(tumor_doses)], -exponent)
    output_whole.fractions_used = np.count_nonzero(~np.isnan(output_whole.physical_doses))

    return output_whole