# -*- coding: utf-8 -*-
"""
whole plan 3D interpolation. This algorithm tracks tumor and OAR BED. If the prescribed tumor dose can be reached, the OAR dose is minimized. If the prescribed tumor dose can not be reached while staying below
maximum BED, the tumor dose is maximized. The value_eval function calculates the optimal dose for one sparing factor given a sparing factor list and the alpha and beta hyperparameter of previous data (can be calculated with data_fit).
the whole_plan function calculates the whole plan given all sparing factors and the hyperparameters.
For extended functions inspect value_eval or whole_plan. Also read the extended function in the readme file.
The value eval function is build by assigning a penalty depending on how much dose has been delivered to the OAR (in each fraction) and by how far we are from the prescribed dose after the last fractions.

"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gamma, truncnorm

# right now once 90 is hit it doesnt seem to matter how much is overdosed. somehow this must be fixed


def data_fit(data):
    """
    This function fits the alpha and beta value for the prior

    Parameters
    ----------
    data : array
        a nxk matrix with n the amount of patints and k the amount of sparing factors per patient.

    Returns
    -------
    list
        alpha and beta hyperparameter.
    """
    std = data.std(axis=1)
    alpha, loc, beta = gamma.fit(std, floc=0)
    return [alpha, beta]


def get_truncated_normal(mean=0, sd=1, low=0.01, upp=10):
    """
    produces a truncated normal distribution

    Parameters
    ----------
    mean : float, optional
        The default is 0.
    sd : float, optional
        The default is 1.
    low : float, optional
        The default is 0.01.
    upp : float, optional
        The default is 10.

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        distribution function.

    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def probdist(X):
    """
    This function produces a probability distribution based on the normal distribution X

    Parameters
    ----------
    X : scipy.stats._distn_infrastructure.rv_frozen
        distribution function.

    Returns
    -------
    prob : list
        list with probabilities for each sparing factor.

    """
    prob = np.zeros(170)
    idx = 0
    for i in np.arange(0.01, 1.71, 0.01):
        prob[idx] = X.cdf(i + 0.004999999999999999999) - X.cdf(i - 0.005)
        idx += 1
    return prob


def std_calc(measured_data, alpha, beta):
    """
    calculates the most likely standard deviation for a list of k sparing factors and a gamma prior
    measured_data: list/array with k sparing factors

    Parameters
    ----------
    measured_data : list/array
        list/array with k sparing factors
    alpha : float
        shape of gamma distribution
    beta : float
        scale of gamma distrinbution

    Returns
    -------
    std : float
        most likely std based on the measured data and gamma prior

    """
    n = len(measured_data)
    std_values = np.arange(0.00001, 0.5, 0.00001)
    likelihood_values = np.zeros(len(std_values))
    for index, value in enumerate(std_values):
        likelihood_values[index] = (
            value ** (alpha - 1)
            / value ** (n - 1)
            * np.exp(-1 / beta * value)
            * np.exp(-np.var(measured_data) / (2 * (value**2 / n)))
        )
    std = std_values[np.argmax(likelihood_values)]
    return std


def argfind(searched_list, value):
    """
    This function is used to find the index of certain values.
    searched_list: list/array with values
    value: value that should be inside the list
    return: index of value

    Parameters
    ----------
    searched_list : list/array
        list in which our searched value is.
    value : float
        item inside list.

    Returns
    -------
    index : integer
        index of value inside list.

    """
    index = min(range(len(searched_list)), key=lambda i: abs(searched_list[i] - value))
    return index


def BED_calc0(dose, ab, sparing=1):
    """
    calculates the BED for a specific dose

    Parameters
    ----------
    dose : float
        physical dose to be delivered.
    ab : float
        alpha-beta ratio of tissue.
    sparing : float, optional
        sparing factor. The default is 1 (tumor).

    Returns
    -------
    BED : float
        BED to be delivered based on dose, sparing factor and alpha-beta ratio.

    """
    BED = sparing * dose * (1 + (sparing * dose) / ab)
    return BED


def BED_calc_matrix(actionspace, ab, sf):
    """
    calculates the BED for an array of values

    Parameters
    ----------
    sf : list/array
        list of sparing factors to calculate the correspondent BED.
    ab : float
        alpha-beta ratio of tissue.
    actionspace : list/array
        doses to be delivered.

    Returns
    -------
    BED : List/array
        list of all future BEDs based on the delivered doses and sparing factors.

    """
    BED = np.outer(sf, actionspace) * (
        1 + np.outer(sf, actionspace) / ab
    )  # produces a sparing factors x actions space array
    return BED


def value_eval(
    fraction,
    number_of_fractions,
    BED_OAR,
    BED_tumor,
    sparing_factors,
    abt,
    abn,
    bound_OAR,
    bound_tumor,
    alpha,
    beta,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    Calculates the optimal dose for the desired fraction.
    fraction: number of actual fraction (1 for first, 2 for second, etc.)
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used
    Parameters
    ----------
    fraction : integer
        number of actual fraction (1 for first, 2 for second, etc.).
    number_of_fractions : integer
        number of fractions that will be delivered.
    BED_OAR : float
        accumulated BED in OAR (from previous fractions) zero in fraction 1.
    BED_tumor : float
        accumulated BED in tumor (from previous fractions) zero in fraction 1.
    sparing_factors : TYPE
        list or array of all sparing factors that have been observed. e.g. list of 3 sparing factors in fraction 2 (planning,fx1,fx2).
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
    bound_OAR : float
        maximal BED of OAR
    bound_tumor : float
        prescribed tumor BED.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients
    min_dose : float
        minimal physical doses to be delivered in one fraction. The doses are aimed at PTV 95
    max_dose : float
        maximal physical doses to be delivered in one fraction. The doses are aimed at PTV 95
    fixed_prob : int
        this variable is to turn on a fixed probability distribution. If the variable is not used (0), then the probability will be updated. If the variable is turned to 1, the inserted mean and std will be used for a fixed sparing factor distribution
    fixed_mean: float
        mean of the fixed sparing factor normal distribution
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution
    Returns
    -------
    list
        list with following arrays/values:
        physical_dose: optimal physical dose for actual fraction
        accumulated_tumor_dose: accumulated tumor BED
        accumulated_OAR_dose: accumulated OAR BED
        tumor_dose: tumor BED to be delivered
        OAR_dose: OAR BED to be delivered

    """

    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_calc(sparing_factors, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = get_truncated_normal(mean=mean, sd=standard_deviation, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf = np.arange(0.01, 1.71, 0.01)
    sf = sf[prob > 0.00001]  # get rid of all probabilities below 10^-5
    prob = prob[prob > 0.00001]
    underdosepenalty = 10
    BEDT = np.arange(BED_tumor, bound_tumor, 1)  # tumordose
    BEDNT = np.arange(BED_OAR, bound_OAR, 1)  # OAR dose
    BEDNT = np.concatenate((BEDNT, [bound_OAR, bound_OAR + 1]))
    BEDT = np.concatenate((BEDT, [bound_tumor, bound_tumor + 1]))
    Values = np.zeros(
        [(number_of_fractions - fraction), len(BEDT), len(BEDNT), len(sf)]
    )  # 2d values list with first indice being the BED and second being the sf
    if max_dose > (-1 + np.sqrt(1 + 4 * 1 * (bound_tumor) / abt)) / (
        2 * 1**2 / abt
    ):  # we constrain the maximum dose so that no more dose than what is needed would be checked in the actionspace
        max_dose = (-1 + np.sqrt(1 + 4 * 1 * (bound_tumor) / abt)) / (2 * 1**2 / abt)
    if min_dose > max_dose:
        min_dose = max_dose - 0.1
    actionspace = np.arange(min_dose, max_dose + 0.1, 0.1)
    policy = np.zeros(
        ((number_of_fractions - fraction), len(BEDT), len(BEDNT), len(sf))
    )
    upperbound_normal_tissue = bound_OAR + 1
    upperbound_tumor = bound_tumor + 1

    OAR_dose = BED_calc_matrix(
        actionspace, abn, sf
    )  # calculates the dose that is deposited into the normal tissue for all sparing factors
    tumor_dose = BED_calc_matrix(actionspace, abt, 1)[
        0
    ]  # this is the dose delivered to the tumor
    actual_fraction_sf = argfind(sf, np.round(sparing_factors[-1], 2))

    for index, frac_state_plus in enumerate(
        np.arange(number_of_fractions + 1, fraction, -1)
    ):  # We have five fractionations with 2 special cases 0 and 4
        frac_state = frac_state_plus - 1
        if (
            frac_state == 1
        ):  # first state with no prior dose delivered so we dont loop through BEDNT
            future_OAR = BED_OAR + OAR_dose[actual_fraction_sf]
            future_tumor = BED_tumor + tumor_dose
            future_OAR[
                future_OAR > bound_OAR
            ] = upperbound_normal_tissue  # any dose surpassing the upper bound will be set to the upper bound which will be penalized strongly
            future_tumor[future_tumor > bound_tumor] = upperbound_tumor
            future_values_prob = (Values[index - 1] * prob).sum(
                axis=2
            )  # future values of tumor and oar state
            value_interpolation = RegularGridInterpolator(
                (BEDT, BEDNT), future_values_prob
            )
            future_value_actual = value_interpolation(
                np.array([future_tumor, future_OAR]).T
            )
            Vs = future_value_actual - OAR_dose[actual_fraction_sf]
            actual_policy = Vs.argmax(axis=0)

        elif (
            frac_state == fraction
        ):  # if we are in the actual fraction we do not need to check all possible BED states but only the one we are in
            if fraction != number_of_fractions:
                future_OAR = BED_OAR + OAR_dose[actual_fraction_sf]
                overdosing = (future_OAR - bound_OAR).clip(min=0)
                future_OAR[
                    future_OAR > bound_OAR
                ] = upperbound_normal_tissue  # any dose surpassing the upper bound will be set to the upper bound which will be penalized strongly
                future_tumor = BED_tumor + tumor_dose
                future_tumor[future_tumor > bound_tumor] = upperbound_tumor
                future_values_prob = (Values[index - 1] * prob).sum(
                    axis=2
                )  # future values of tumor and oar state
                value_interpolation = RegularGridInterpolator(
                    (BEDT, BEDNT), future_values_prob
                )
                penalties = (
                    overdosing * -10000000000
                )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                future_value_actual = value_interpolation(
                    np.array([future_tumor, future_OAR]).T
                )
                Vs = future_value_actual - OAR_dose[actual_fraction_sf] + penalties
                actual_policy = Vs.argmax(axis=0)
            else:
                sf_end = sparing_factors[-1]
                best_action_BED = (
                    -sf_end
                    + np.sqrt(
                        sf_end**2 + 4 * sf_end**2 * (bound_OAR - BED_OAR) / abn
                    )
                ) / (2 * sf_end**2 / abn)
                best_action_tumor = (
                    -1 + np.sqrt(1 + 4 * 1 * (bound_tumor - BED_tumor) / abt)
                ) / (2 * 1**2 / abt)
                best_action = np.min([best_action_BED, best_action_tumor], axis=0)
                if BED_OAR > bound_OAR or BED_tumor > bound_tumor:
                    best_action = np.ones(best_action.shape) * min_dose
                if best_action > max_dose:
                    best_action = max_dose
                if best_action < min_dose:
                    best_action = min_dose

                future_tumor = BED_tumor + BED_calc0(best_action, abt)
                future_OAR = BED_OAR + BED_calc0(best_action, abn, sf_end)
                actual_policy = best_action * 10
        else:
            future_values_prob = (Values[index - 1] * prob).sum(
                axis=2
            )  # future values of tumor and oar state
            value_interpolation = RegularGridInterpolator(
                (BEDT, BEDNT), future_values_prob
            )  # interpolation function
            for tumor_index, tumor_value in enumerate(BEDT):
                for OAR_index, OAR_value in enumerate(
                    BEDNT
                ):  # this and the next for loop allow us to loop through all states
                    future_OAR = OAR_dose + OAR_value
                    overdosing = (future_OAR - bound_OAR).clip(min=0)
                    future_OAR[
                        future_OAR > bound_OAR
                    ] = upperbound_normal_tissue  # any dose surpassing 90.1 is set to 90.1
                    future_tumor = tumor_value + tumor_dose
                    future_tumor[
                        future_tumor > bound_tumor
                    ] = upperbound_tumor  # any dose surpassing the tumor bound is set to tumor_bound + 0.1

                    if (
                        frac_state == number_of_fractions
                    ):  # last state no more further values to add
                        best_action_BED = (
                            -sf
                            + np.sqrt(
                                sf**2 + 4 * sf**2 * (bound_OAR - OAR_value) / abn
                            )
                        ) / (
                            2 * sf**2 / abn
                        )  # calculate maximal dose that can be delivered to OAR and tumor
                        best_action_tumor = (
                            -np.ones(len(sf))
                            + np.sqrt(
                                np.ones(len(sf))
                                + 4
                                * np.ones(len(sf))
                                * (bound_tumor - tumor_value)
                                / abt
                            )
                        ) / (2 * np.ones(len(sf)) ** 2 / abt)
                        best_action = np.min(
                            [best_action_BED, best_action_tumor], axis=0
                        )  # take the smaller of both doses to not surpass the limit
                        best_action[best_action > max_dose] = max_dose
                        best_action[best_action < min_dose] = min_dose
                        if (
                            OAR_value > bound_OAR or tumor_value > bound_tumor
                        ):  # if the limit is already surpassed we add a penaltsy
                            best_action = np.ones(best_action.shape) * min_dose
                        future_OAR = OAR_value + BED_calc0(best_action, abn, sf)
                        future_tumor = tumor_value + BED_calc0(best_action, abt, 1)
                        overdose_penalty2 = np.zeros(
                            best_action.shape
                        )  # we need a second penalty if we overdose in the last fraction
                        overdose_penalty3 = np.zeros(best_action.shape)
                        overdose_penalty2[
                            future_tumor > bound_tumor + 0.0001
                        ] = -100000000000
                        overdose_penalty3[
                            future_OAR > bound_OAR + 0.0001
                        ] = (
                            -100000000000
                        )  # A small number has to be added as sometimes 90. > 90 was True
                        end_penalty = (
                            -abs(future_tumor - bound_tumor) * underdosepenalty
                        )  # the farther we are away from the prescribed dose, the higher the penalty. Under- and overdosing is punished
                        end_penalty_OAR = (
                            -(future_OAR - bound_OAR).clip(min=0) * 1000
                        )  # if overdosing the OAR is not preventable, the overdosing should stay as low as possible
                        Values[index][tumor_index][OAR_index] = (
                            end_penalty
                            - BED_calc0(best_action, abn, sf)
                            + overdose_penalty2
                            + overdose_penalty3
                            + end_penalty_OAR
                        )  # we also substract all the dose delivered to the OAR so the algorithm tries to minimize it
                        policy[index][tumor_index][OAR_index] = best_action * 10
                    else:
                        future_value = np.zeros([len(sf), len(actionspace)])
                        for actual_sf in range(0, len(sf)):
                            future_value[actual_sf] = value_interpolation(
                                np.array([future_tumor, future_OAR[actual_sf]]).T
                            )
                        penalties = (
                            overdosing * -10000000000
                        )  # additional penalty when overdosing is needed when choosing a minimum dose to be delivered
                        Vs = future_value - OAR_dose + penalties
                        best_action = Vs.argmax(axis=1)
                        valer = Vs.max(axis=1)
                        policy[index][tumor_index][OAR_index] = best_action
                        Values[index][tumor_index][OAR_index] = valer
    if fraction != number_of_fractions:
        physical_dose = actionspace[actual_policy]
    else:
        physical_dose = actual_policy / 10
    tumor_dose = BED_calc0(physical_dose, abt)
    OAR_dose = BED_calc0(physical_dose, abn, sparing_factors[-1])
    accumulated_tumor_dose = BED_calc0(physical_dose, abt) + BED_tumor
    accumulated_OAR_dose = BED_calc0(physical_dose, abn, sparing_factors[-1]) + BED_OAR
    return [
        physical_dose,
        accumulated_tumor_dose,
        accumulated_OAR_dose,
        tumor_dose,
        OAR_dose,
    ]


def whole_plan(
    number_of_fractions,
    sparing_factors,
    abt,
    abn,
    bound_OAR,
    bound_tumor,
    alpha,
    beta,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    std_fixed=0,
):
    """
    calculates all doses for a number_of_fractions fraction treatment (with 6 known sparing factors)
    sparing_factors: list or array of number_of_fractions + 1 sparing factors that have been observed.
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used
    Parameters
    ----------
    sparing_factors : list/array
        list or array of 6 sparing factors that have been observed.
    number_of_fractions : integer
        number of fractions that will be delivered.
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
    bound_OAR : float
        maximal BED of OAR.
    bound_tumor : float
        prescribed tumor BED.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    min_dose : float
        minimal physical doses to be delivered in one fraction. The doses are aimed at PTV 95
    max_dose : float
        maximal physical doses to be delivered in one fraction. The doses are aimed at PTV 95
    fixed_prob : int
        this variable is to turn on a fixed probability distribution. If the variable is not used (0), then the probability will be updated. If the variable is turned to 1, the inserted mean and std will be used for a fixed sparing factor distribution
    fixed_mean: float
        mean of the fixed sparing factor normal distribution
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution
    Returns
    -------
    List with delivered tumor doses, delivered OAR doses and delivered physical doses

    """
    physical_doses = np.zeros(number_of_fractions)
    tumor_doses = np.zeros(number_of_fractions)
    OAR_doses = np.zeros(number_of_fractions)
    accumulated_OAR_dose = 0
    accumulated_tumor_dose = 0
    for looper in range(0, number_of_fractions):
        [
            actual_policy,
            accumulated_tumor_dose,
            accumulated_OAR_dose,
            tumor_dose,
            OAR_dose,
        ] = value_eval(
            looper + 1,
            number_of_fractions,
            accumulated_OAR_dose,
            accumulated_tumor_dose,
            sparing_factors[0 : looper + 2],
            abt,
            abn,
            bound_OAR,
            bound_tumor,
            alpha,
            beta,
            min_dose,
            max_dose,
            fixed_prob,
            fixed_mean,
            std_fixed,
        )
        physical_doses[looper] = actual_policy
        tumor_doses[looper] = tumor_dose
        OAR_doses[looper] = OAR_dose
        print("calcultions fraction ", looper + 1, " done")
    return [tumor_doses, OAR_doses, physical_doses]


def whole_plan_print(
    number_of_fractions,
    sparing_factors,
    abt,
    abn,
    bound_OAR,
    bound_tumor,
    alpha,
    beta,
    min_dose=0,
    max_dose=22.2,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    calculates all doses for a 5 fraction treatment (with 6 known sparing factors)
    sparing_factors: list or array of 6 sparing factors that have been observed.
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used. This function prints the result and is optimal to be used when working with the script instead of the GUI
    Parameters
    ----------
    sparing_factors : list/array
        list or array of 6 sparing factors that have been observed.
    number_of_fractions : integer
        number of fractions that will be delivered.
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
    bound_OAR : float
        maximal BED of OAR.
    bound_tumor : float
        prescribed tumor BED.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    min_dose : float
        minimal physical doses to be delivered in one fraction. The doses are aimed at PTV 95
    max_dose : float
        maximal physical doses to be delivered in one fraction. The doses are aimed at PTV 95
    fixed_prob : int
        this variable is to turn on a fixed probability distribution. If the variable is not used (0), then the probability will be updated. If the variable is turned to 1, the inserted mean and std will be used for a fixed sparing factor distribution
    fixed_mean: float
        mean of the fixed sparing factor normal distribution
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution
    Returns
    -------
    None.

    """
    [tumor_doses, OAR_doses, physical_doses] = whole_plan(
        number_of_fractions,
        sparing_factors,
        abt,
        abn,
        bound_OAR,
        bound_tumor,
        alpha,
        beta,
        min_dose,
        max_dose,
        fixed_prob,
        fixed_mean,
        fixed_std,
    )
    for i in range(number_of_fractions):
        print("fraction ", i + 1)
        print("physical dose delivered = ", np.round(physical_doses[i], 2))
        print("tumor BED delivered = ", np.round(tumor_doses[i], 2))
        print("OAR BED delivered = ", np.round(OAR_doses[i], 2))
    print("total tumor BED = ", np.sum(tumor_doses))
    print("total OAR BED = ", np.sum(OAR_doses))


def single_fraction_print(
    number_of_fractions,
    sparing_factors,
    BED_OAR,
    BED_tumor,
    abt,
    abn,
    bound_OAR,
    bound_tumor,
    alpha,
    beta,
    min_dose,
    max_dose,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    calculates the actual dose for a number_of_fractions fraction treatment
    sparing_factors: list or array with all sparing factors that have been observed.
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used
    Parameters
    ----------
    sparing_factors : list/array
        list or array of all sparing factors that have been observed. e.g. list of 3 sparing factors in fraction 2 (planning,fx1,fx2).
    number_of_fractions : integer
        number of fractions that will be delivered.
    BED_OAR : float
        accumulated BED in OAR (from previous fractions) zero in fraction 1.
    BED_tumor : float
        accumulated BED in tumor (from previous fractions) zero in fraction 1.
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
    bound_OAR : float
        maximal BED of OAR
    bound_tumor : float
        prescribed tumor BED.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    min_dose : float
        minimal physical doses to be delivered in one fraction. The doses are aimed at PTV 95.
    max_dose : float
        maximal physical doses to be delivered in one fraction. The doses are aimed at PTV 95 .
    fixed_prob : int
        this variable is to turn on a fixed probability distribution. If the variable is not used (0), then the probability will be updated. If the variable is turned to 1, the inserted mean and std will be used for a fixed sparing factor distribution
    fixed_mean: float
        mean of the fixed sparing factor normal distribution
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution
    Returns
    -------
    None.

    """
    [
        actual_policy,
        accumulated_tumor_dose,
        accumulated_OAR_dose,
        tumor_dose,
        OAR_dose,
    ] = value_eval(
        len(sparing_factors) - 1,
        number_of_fractions,
        BED_OAR,
        BED_tumor,
        sparing_factors,
        abt,
        abn,
        bound_OAR,
        bound_tumor,
        alpha,
        beta,
        min_dose,
        max_dose,
        fixed_prob,
        fixed_mean,
        fixed_std,
    )
    print("fraction ", len(sparing_factors) - 1)
    print("physical dose delivered = ", actual_policy)
    print("tumor BED delivered = ", tumor_dose)
    print("OAR BED delivered = ", OAR_dose)
