# -*- coding: utf-8 -*-
"""
interpolation program for OAR minimisation whole plan with minimum and maximum physical dose (to tumor)
2D interpolation. In this program the optimal fraction doses are compueted based on a prescribed tumor dose while minimizing OAR BED.
single_fraction allows to compute single fraction doses, while whole_plan computes the doses for a whole treatment plan (when all sparing factors are known).
whole_plan_print prints the doses in a well-aranged manner.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma, truncnorm


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
    standard_devs = data.std(axis=1)
    alpha, loc, beta = gamma.fit(standard_devs, floc=0)
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
    calculates the most likely standard deviation for a list of k sparing factors and a gamma conjugate prior
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
        )  # here i have to check whether it is good.
    std = std_values[np.argmax(likelihood_values)]
    return std


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


def BED_calc_matrix(sf, ab, actionspace):
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


def max_action(bed, actionspace, goal, abt=10):
    """
    Computes the maximal dose that can be delivered to the tumor in each fraction depending on the actual accumulated dose

    Parameters
    ----------
    bed : float
        accumulated tumor dose so far.
    actionspace : list/array
        array with all discrete dose steps.
    goal : float
        prescribed tumor dose.
    abt : float, optional
        alpha beta ratio of tumor. The default is 10.

    Returns
    -------
    sizer : integer
        gives the size of the resized actionspace to reach the prescribed tumor dose.

    """
    BED_action = BED_calc0(actionspace, abt)
    max_action = min(max(BED_action), goal - bed)
    sizer = np.argmin(np.abs(BED_action - max_action))
    return sizer


def argfind(BEDT, value):
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
    index = min(range(len(BEDT)), key=lambda i: abs(BEDT[i] - value))
    return index


def value_eval(
    fraction,
    number_of_fractions,
    accumulated_dose,
    sparing_factors,
    alpha,
    beta,
    goal,
    abt,
    abn,
    C,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    calculates the optimal dose for the desired fraction.

    Parameters
    ----------
    fraction : integer
        Number of the actual fraction.
    number_of_fractions : integer
        number of fractions that will be delivered.
    accumulated_dose : float
        accumulated tumor BED.
    sparing_factors : list/array
        list or array of all observed sparing factors. Include planning session sparing factor!.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients
    goal : float
        prescribed tumor BED.
    abt : float, optional
        alpha beta ratio of the tumor. The default is 10.
    abn : float, optional
        alpha beta ratio of the organ at risk. The default is 3.
    C: float
        fixed constant to penalize for each additional fraction that is used
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
        Returns list with policies,relevant sparing factors range, physical dose to be delivered, tumor BED to be delivered, OAR BED to be delivered.

    """
    if fixed_prob != 1:
        mean = np.mean(
            sparing_factors
        )  # extract the mean and std to setup the sparingfactor distribution
        standard_deviation = std_calc(sparing_factors, alpha, beta)
    if fixed_prob == 1:
        mean = fixed_mean
        standard_deviation = fixed_std
    X = get_truncated_normal(mean=mean, sd=standard_deviation, low=0.01, upp=1.3)
    BEDT = np.arange(accumulated_dose, goal, 1)
    BEDT = np.concatenate(
        (BEDT, [goal, goal + 1])
    )  # add an extra step outside of our prescribed tumor dose which will be penalized to make sure that we aim at the prescribe tumor dose
    prob = np.array(probdist(X))
    sf = np.arange(0.01, 1.71, 0.01)
    sf = sf[prob > 0.00001]
    prob = prob[prob > 0.00001]
    # we prepare an empty values list and open an action space which is equal to all the dose numbers that can be given in one fraction
    Values = np.zeros(
        (number_of_fractions - fraction, len(BEDT), len(sf))
    )  # 2d values list with first indice being the BED and second being the sf
    if max_dose > (-1 + np.sqrt(1**2 + 4 * 1**2 * (goal) / abt)) / (
        2 * 1**2 / abt
    ):  # if the max dose is too large we lower it, so we dont needlessly check too many actions
        max_dose = np.round(
            (-1 + np.sqrt(1**2 + 4 * 1**2 * (goal) / abt)) / (2 * 1**2 / abt), 2
        )
    if min_dose > max_dose:
        min_dose = max_dose - 0.1
    actionspace = np.arange(min_dose, max_dose + 0.11, 0.1) #set +0.11 to ensure at least max_dose is covered
    # now we set up the policy array which has len(BEDT)*len(sf)*len(actionspace) entries. We give each action the same probability to start with
    policy = np.zeros((number_of_fractions - fraction, len(BEDT), len(sf)))

    for state, fraction_state in enumerate(
        np.arange(number_of_fractions, fraction -1, -1)
    ):
        if (
            state == number_of_fractions - 1 #fraction_state == 1
        ):  # first state with no prior dose delivered so we dont loop through BEDT
            BEDN = BED_calc_matrix(
                sparing_factors[-1], abn, actionspace
            )  # calculate all delivered doses to the Normal tissues (the penalty)
            future_BEDT = BED_calc0(actionspace, abt)
            future_values_func = interp1d(BEDT, (Values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_BEDT
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            C_penalties = np.zeros(future_BEDT.shape)
            C_penalties[future_BEDT < goal] = -C
            Vs = -BEDN + C_penalties + future_values
            policy4 = Vs.argmax(axis=1)
        elif (
            accumulated_dose >= goal
        ):
            best_action = 0
            last_BEDN = BED_calc0(best_action, abn, sparing_factors[-1])
            policy4 = 0
            break
        elif (
            fraction_state == fraction and fraction != number_of_fractions
        ):  # actual fraction is first state to calculate
            actionspace_clipped = actionspace[
                0 : max_action(accumulated_dose, actionspace, goal) + 1
            ]
            BEDN = BED_calc_matrix(sparing_factors[-1], abn, actionspace_clipped)
            future_BEDT = accumulated_dose + BED_calc0(actionspace_clipped, abt)
            future_BEDT[future_BEDT > goal] = goal + 1
            penalties = np.zeros(future_BEDT.shape)
            C_penalties = np.zeros(future_BEDT.shape)
            penalties[future_BEDT > goal] = 0 #overdosing is indirectly penalised with BEDN
            C_penalties[future_BEDT < goal] = -C
            future_values_func = interp1d(BEDT, (Values[state - 1] * prob).sum(axis=1))
            future_values = future_values_func(
                future_BEDT
            )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
            Vs = -BEDN + C_penalties + future_values + penalties
            policy4 = Vs.argmax(axis=1)
        elif (
            fraction == number_of_fractions
        ):  # in this state no penalty has to be defined as the value is not relevant
            best_action = (
                -1 + np.sqrt(1 + 4 * 1 * (goal - accumulated_dose) / abt)
            ) / (2 * 1**2 / abt)
            if accumulated_dose > goal:
                best_action = 0
            if best_action < min_dose:
                best_action = min_dose
            if best_action > max_dose:
                best_action = max_dose
            last_BEDN = BED_calc0(best_action, abn, sparing_factors[-1])
            policy4 = best_action * 10
        else:
            future_value_prob = (Values[state - 1] * prob).sum(axis=1)
            future_values_func = interp1d(BEDT, future_value_prob)
            for tumor_index, tumor_value in enumerate(
                BEDT
            ):  # this and the next for loop allow us to loop through all states
                actionspace_clipped = actionspace[
                    0 : max_action(tumor_value, actionspace, goal) + 1
                ]  # we only allow the actions that do not overshoot
                BEDN = BED_calc_matrix(
                    sf, abn, actionspace_clipped
                )  # this one could be done outside of the loop and only the clipping would happen inside the loop.
                BED = BED_calc_matrix(np.ones(len(sf)), abt, actionspace_clipped)
                if state != 0 and tumor_value < goal:
                    future_BEDT = tumor_value + BED
                    future_BEDT[future_BEDT > goal] = goal + 1
                    future_values = future_values_func(
                        future_BEDT
                    )  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                    C_penalties = np.zeros(future_BEDT.shape)
                    C_penalties[future_BEDT < goal] = -C
                    Vs = -BEDN + C_penalties + future_values
                    if Vs.size == 0:
                        best_action = np.zeros(len(sf))
                        valer = np.zeros(len(sf))
                    else:
                        best_action = Vs.argmax(axis=1)
                        valer = Vs.max(axis=1)
                elif tumor_value > goal: #calculate value for terminal case
                    best_action = 0
                    future_BEDT = tumor_value
                    valer = (
                        + np.zeros(sf.shape)
                    )  
                else:  # last state no more further values to add
                    best_action = (
                        -1 + np.sqrt(1 + 4 * 1 * (goal - tumor_value) / abt)
                    ) / (2 * 1**2 / abt)
                    if best_action > max_dose:
                        best_action = max_dose
                    if best_action < min_dose:
                        best_action = min_dose
                    last_BEDN = BED_calc0(best_action, abn, sf)
                    future_BEDT = tumor_value + BED_calc0(best_action, abt)
                    underdose_penalty = 0
                    overdose_penalty = 0
                    if future_BEDT < goal:
                        underdose_penalty = -10000
                    if future_BEDT > goal:
                        overdose_penalty = 0
                    valer = (
                        -last_BEDN
                        + underdose_penalty * np.ones(sf.shape)
                        + overdose_penalty * np.ones(sf.shape)
                    )  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for

                policy[state][tumor_index] = best_action
                Values[state][tumor_index] = valer
    if fraction != number_of_fractions:
        physical_dose = actionspace[policy4]
    if fraction == number_of_fractions:
        physical_dose = policy4 / 10
    tumor_dose = BED_calc0(physical_dose, abt)
    OAR_dose = BED_calc0(physical_dose, abn, sparing_factors[-1])
    return [policy, sf, physical_dose, tumor_dose, OAR_dose]


def whole_plan(
    number_of_fractions,
    sparing_factors,
    alpha,
    beta,
    goal,
    C,
    abt=10,
    abn=3,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    calculates all doses for a number_of_fractions fraction treatment (with 6 known sparing factors)
    sparing_factors: list or array of number_of_fractions + 1 sparing factors that have been observed.
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used
    Parameters
    ----------
    number_of_fractions : integer
        number of fractions that will be delivered.
    sparing_factors : list/array
        list or array of 6 sparing factors that have been observed.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    goal : float
        prescribed tumor BED
    C: float
        fixed constant to penalize for each additional fraction that is used
    abt : float
        alpha-beta ratio of tumor. default is 10
    abn : float
        alpha-beta ratio of OAR. default is 3
    min_dose : float
        minimal physical doses to be delivered in one fraction. The doses are aimed at PTV 95. Defaut is 0
    max_dose : float
        maximal physical doses to be delivered in one fraction. The doses are aimed at PTV 95. Default is 22.3
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
    accumulated_tumor_dose = 0
    accumulated_OAR_dose = 0
    physical_doses = np.zeros(number_of_fractions)
    tumor_doses = np.zeros(number_of_fractions)
    OAR_doses = np.zeros(number_of_fractions)
    for looper in range(0, number_of_fractions):
        [policy, sf, physical_dose, tumor_dose, OAR_dose] = value_eval(
            looper + 1,
            number_of_fractions,
            accumulated_tumor_dose,
            sparing_factors[0 : looper + 2],
            alpha,
            beta,
            goal,
            abt,
            abn,
            C,
            min_dose,
            max_dose,
            fixed_prob,
            fixed_mean,
            fixed_std,
        )
        accumulated_tumor_dose += tumor_dose
        accumulated_OAR_dose += OAR_dose
        tumor_doses[looper] = tumor_dose
        OAR_doses[looper] = OAR_dose
        physical_doses[looper] = physical_dose
    return [tumor_doses, OAR_doses, physical_doses]


def whole_plan_print(
    number_of_fractions,
    sparing_factors,
    alpha,
    beta,
    goal,
    abt=10,
    abn=3,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):
    """
    calculates all doses for a number_of_fractions fraction treatment (with 6 known sparing factors)
    sparing_factors: list or array of number_of_fractions + 1 sparing factors that have been observed.
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used
    Parameters. This function prints the result and is optimal to be used when working with the script instead of the GUI
    ----------
    number_of_fractions : integer
        number of fractions that will be delivered.
    sparing_factors : list/array
        list or array of 6 sparing factors that have been observed.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    goal : float
        prescribed tumor BED
    abt : float
        alpha-beta ratio of tumor. default is 10
    abn : float
        alpha-beta ratio of OAR. default is 3
    min_dose : float
        minimal physical doses to be delivered in one fraction. The doses are aimed at PTV 95. Defaut is 0
    max_dose : float
        maximal physical doses to be delivered in one fraction. The doses are aimed at PTV 95. Default is 22.3
    fixed_prob : int
        this variable is to turn on a fixed probability distribution. If the variable is not used (0), then the probability will be updated. If the variable is turned to 1, the inserted mean and std will be used for a fixed sparing factor distribution
    fixed_mean: float
        mean of the fixed sparing factor normal distribution
    std_fixed: float
        standard deviation of the fixed sparing factor normal distribution
    Returns
    -------
    None

    """
    [tumor_doses, OAR_doses, physical_doses] = whole_plan(
        number_of_fractions,
        sparing_factors,
        alpha,
        beta,
        goal,
        abt,
        abn,
        min_dose,
        max_dose,
        fixed_prob,
        fixed_mean,
        fixed_std,
    )
    for i in range(len(physical_doses)):
        print("physical dose delivered in fraction ", i + 1, "  = ", physical_doses[i])
        print("tumor BED", tumor_doses[i])
        print("OAR BED", OAR_doses[i])
    print("accumulated tumor dose: ", np.sum(tumor_doses))
    print("accumulated OAR dose: ", np.sum(OAR_doses))


def single_fraction(
    number_of_fractions,
    sparing_factors,
    accumulated_tumor_dose,
    alpha,
    beta,
    goal,
    abt=10,
    abn=3,
    min_dose=0,
    max_dose=22.3,
    fixed_prob=0,
    fixed_mean=0,
    fixed_std=0,
):  # Vervollständigen
    """
    calculates the actual dose for a number_of_fractions fraction treatment
    sparing_factors: list or array with all sparing factors that have been observed.
    To used a fixed probability distribution, change fixed_prob to 1 and add a fixed mean and std.
    If a fixed probability distribution is chosen, alpha and beta are to be chosen arbitrarily as they will not be used
    Parameters
    ----------
    number_of_fractions : integer
        number of fractions that will be delivered.
    sparing_factors : list/array
        list or array of all sparing factors that have been observed. e.g. list of 3 sparing factors in fraction 2 (planning,fx1,fx2).
    accumulated_tumor_dose : float
        accumulated BED in tumor (from previous fractions) zero in fraction 1.
    alpha : float
        alpha hyperparameter of std prior derived from previous patients.
    beta : float
        beta hyperparameter of std prior derived from previous patients.
    goal: float
        prescribed tumor dose
    abt : float
        alpha-beta ratio of tumor.
    abn : float
        alpha-beta ratio of OAR.
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
    [policy, sf, physical_dose, tumor_dose, OAR_dose] = value_eval(
        len(sparing_factors) - 1,
        number_of_fractions,
        accumulated_tumor_dose,
        sparing_factors,
        alpha,
        beta,
        goal,
        abt,
        abn,
        min_dose,
        max_dose,
        fixed_prob,
        fixed_mean,
        fixed_std,
    )
    print(
        "physical dose delivered in fraction ",
        len(sparing_factors) - 1,
        ": ",
        physical_dose,
    )
    print("tumor BED: ", tumor_dose)
    print("OAR BED", OAR_dose)
