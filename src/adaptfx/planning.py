# -*- coding: utf-8 -*-
"""
This function computes the doses for a whole
treatment plan (when all sparing factors are known)

or for a single fraction with known previous
sparing factors and accumulated BED
"""

import numpy as np
import adaptfx as afx

def multiple(algorithm, keys, settings=afx.SETTING_DICT):
    """
    calculates whole plan given all sparing factors

    General Parameters
    ----------
    number_of_fractions : integer
        number of fractions that will be delivered.
    sparing_factors : list/array
        list/array with all observed sparing factors.
    alpha : float
        shape of inverse-gamma distribution.
    beta : float
        scale of inverse-gamme distrinbution.
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
        If the variable is not used (0), then the probability will be updated.
        If the variable is turned to 1, the inserted mean and std will be used
        for a fixed sparing factor distribution. Then alpha and beta are unused.
    fixed_mean: float
        mean of the fixed sparing factor normal distribution.
    fixed_std: float
        standard deviation of the fixed sparing factor normal distribution.

    Specific Parameters
    ----------
    oar_limit : float
        upper BED limit of OAR
    tumor_goal : float
        prescribed tumor BED.
    C: float
        fixed constant to penalise for each additional fraction that is used.

    Returns
    -------
    list

    """
    if isinstance(keys, dict):
        # check if keys is a dictionary from manual user
        keys = afx.DotDict(keys)
    
    # k.tumor_goal=keys.tumor_goal
    # k.oar_limit=keys.oar_limit
    # k.c constant

    physical_doses = np.zeros(keys.number_of_fractions)
    tumor_doses = np.zeros(keys.number_of_fractions)
    oar_doses = np.zeros(keys.number_of_fractions)

    for i in range(0, keys.number_of_fractions):
        keys.fraction = i + 1
        keys.accumulated_tumor_dose = tumor_doses.sum()
        keys.sparing_factors_public = keys.sparing_factors[0 : i + 2]
        if algorithm == 'oar':
            [
                physical_doses[i],
                tumor_doses[i],
                oar_doses[i]
            ] = afx.min_oar_bed(
                keys,
                settings,
            )
        # elif algorithm == 'tumor':
        #     [
        #         physical_dose,
        #         tumor_dose,
        #         oar_dose,
        #     ] = tumor.value_eval(
        #         looper + 1,
        #         number_of_fractions,
        #         accumulated_oar_dose,
        #         sparing_factors[: looper + 2],
        #         alpha,
        #         beta,
        #         oar_limit,
        #         abt,
        #         abn,
        #         min_dose,
        #         max_dose,
        #         fixed_prob,
        #         fixed_mean,
        #         fixed_std,
        #     )
        # elif algorithm == 'frac':
        #     [
        #         physical_dose,
        #         tumor_dose,
        #         oar_dose
        #     ] = frac.value_eval(
        #         looper + 1,
        #         number_of_fractions,
        #         accumulated_tumor_dose,
        #         sparing_factors[0 : looper + 2],
        #         alpha,
        #         beta,
        #         tumor_goal,
        #         abt,
        #         abn,
        #         C,
        #         min_dose,
        #         max_dose,
        #         fixed_prob,
        #         fixed_mean,
        #         fixed_std,
        #     )

        # elif algorithm == 'tumor_oar':
        #     [
        #         physical_dose,
        #         tumor_dose,
        #         oar_dose
        #     ] = tumor_oar.value_eval(
        #         looper + 1,
        #         number_of_fractions,
        #         accumulated_oar_dose,
        #         accumulated_tumor_dose,
        #         sparing_factors[0 : looper + 2],
        #         oar_limit,
        #         tumor_goal,
        #         alpha,
        #         beta,
        #         abt,
        #         abn,
        #         min_dose,
        #         max_dose,
        #         fixed_prob,
        #         fixed_mean,
        #         fixed_std,
        #     )

    exponent = afx.find_exponent(settings.dose_stepsize)
    physical, tumor, oar = np.around(
        [physical_doses, tumor_doses, oar_doses], -exponent)
    oar_sum, tumor_sum = np.around(
        [oar_doses.sum(), tumor_doses.sum()], -exponent)

    return [
                oar_sum, tumor_sum, 
                np.array([physical, tumor, oar])
            ]