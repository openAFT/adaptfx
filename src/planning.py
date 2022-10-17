# -*- coding: utf-8 -*-
"""
This function computes the doses for a whole
treatment plan (when all sparing factors are known).
"""

import numpy as np
import fraction_minimisation as frac
import oar_minimisation as oar
# import tumor_maximisation as tumor
# import track_tumor_oar as tumor_oar

def multiple(algorithm, params):
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
    
    number_of_fractions=params['number_of_fractions']
    sparing_factors=params['sparing_factors']
    alpha=params['alpha']
    beta=params['beta']
    tumor_goal=params['tumor_goal']
    oar_limit=params['oar_limit']
    C=params['C']
    abt=params['abt']
    abn=params['abn']
    min_dose=params['min_dose']
    max_dose=params['max_dose']
    fixed_prob=params['fixed_prob']
    fixed_mean=params['fixed_mean']
    fixed_std=params['fixed_std']

    accumulated_tumor_dose = 0
    accumulated_oar_dose = 0
    physical_doses = np.zeros(number_of_fractions)
    tumor_doses = np.zeros(number_of_fractions)
    oar_doses = np.zeros(number_of_fractions)

    # if algorithm == 'frac':
    #     policy_list = []
    #     BEDT_list = []
    #     sf_list = []

    for looper in range(0, number_of_fractions):
        if algorithm == 'oar':
            [
                physical_dose,
                tumor_dose,
                oar_dose
            ] = oar.value_eval(
                looper + 1,
                number_of_fractions,
                accumulated_tumor_dose,
                sparing_factors[0 : looper + 2],
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
        #     # policy_list.append(policy)
        #     # BEDT_list.append(BEDT)
        #     # sf_list.append(sf)

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

        accumulated_tumor_dose += tumor_dose
        accumulated_oar_dose += oar_dose
        physical_doses[looper] = physical_dose
        tumor_doses[looper] = tumor_dose
        oar_doses[looper] = oar_dose

    return [
                accumulated_oar_dose,
                accumulated_tumor_dose,
                np.array((physical_doses,
                tumor_doses,
                oar_doses))
            ]