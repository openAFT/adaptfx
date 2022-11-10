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

    physical_doses = np.zeros(keys.number_of_fractions)
    tumor_doses = np.zeros(keys.number_of_fractions)
    oar_doses = np.zeros(keys.number_of_fractions)

    for i in range(0, keys.number_of_fractions):
        keys.fraction = i + 1
        keys.accumulated_tumor_dose = tumor_doses.sum()
        keys.sparing_factors_public = keys.sparing_factors[0 : i + 2]
        if algorithm == 'oar':
            output = afx.min_oar_bed(keys, sets)
        elif algorithm == 'oar_old':
            output = afx.min_oar_bed_old(keys, sets)
        elif algorithm == 'frac':
            output = afx.min_n_frac(keys, sets)
        elif algorithm == 'frac_old':
            output = afx.min_n_frac_old(keys, sets)
        else:
            afx.aft_error('no valid algorithm given', nme)
            
        physical_doses[i] = output.physical_dose
        tumor_doses[i] = output.tumor_dose
        oar_doses[i] = output.oar_dose

    exponent = afx.find_exponent(sets.dose_stepsize)
    physical, tumor, oar = np.around(
        [physical_doses, tumor_doses, oar_doses], -exponent)
    oar_sum, tumor_sum = np.around(
        [oar_doses.sum(), tumor_doses.sum()], -exponent)

    return [
            oar_sum, tumor_sum,
            np.array([physical, tumor, oar])
            ]