# -*- coding: utf-8 -*-
"""
In this file are all functions to calculate the optimal dose for a single
fraction, given that all previous sparing factors are known and prior date of
patients from the same population is available.

The value_eval function gives the optimal dose for a certain fraction. As input
the sparing factors are needed and the alpha and beta hyperparameter of a
inverse-gamma distribution to improve the probability distribution. If the
alpha and beta value are not known, the data_fit function can be used which
needs the sparing factors of prior patients as input.

The optimal policies can be extracted from pol4 and pol manually (pol4 = first
fraction, first index in pol is the last fraction and the last index is the
first fraction). But one must know which sparing factor is on which index. To
do so one must use the extracted sf from value_eval which tells us which
sparing factors have been used on which index. It is recommended to usethe
result_calc_BEDNT to calculate plans with different sparing factors. This
program uses a discrete state space and does not interpolate between states.
Therefore, it is less precise than the interpolation programs.
"""

import time

import numpy as np
from scipy.stats import invgamma, truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """produces a truncated normal distribution"""
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def probdist(X):
    """
    This function produces a probability distribution based on the normal
    distribution X.
    """
    prob = np.zeros(131)
    idx = 0
    for i in np.arange(0, 1.31, 0.01):
        prob[idx] = X.cdf(i + 0.004999999999999999999) - X.cdf(i - 0.005)
        idx += 1
    return prob


def data_fit(data):
    """
    This function fits the alpha and beta value for the conjugate prior

    data: a nxk matrix with n the amount of patints and k the amount of sparing
    factors per patient.
    """
    variances = data.var(axis=1)
    alpha, loc, beta = invgamma.fit(variances, floc=0)
    return [alpha, beta]


def std_calc(measured_data, alpha, beta):
    """
    Calculate the most likely standard deviation for a list of k sparing
    factors and an inverse-gamma conjugate prior.

    measured_data: list/array with k sparing factors
    alpha: shape of inverse-gamma distribution
    beta: scale of inverse-gamme distrinbution
    return: most likely std based on the measured data and inverse-gamma prior
    """
    n = len(measured_data)
    var_values = np.arange(0.00001, 0.25, 0.00001)
    likelihood_values = np.zeros(len(var_values))
    for index, value in enumerate(var_values):
        likelihood_values[index] = (
            value ** (-alpha - 1)
            / value ** (n / 2)
            * np.exp(-beta / value)
            * np.exp(-np.var(measured_data) * n / (2 * value))
        )
    std = np.sqrt(var_values[np.argmax(likelihood_values)])
    return std


def BED_calc0(dose, ab, sparing=1):
    """calculates the BED for a specific dose"""
    BED = sparing * dose * (1 + (sparing * dose) / ab)
    return BED


def BED_calc(sf, ab, actionspace):
    """calculates the BED for an array of values"""
    BED = np.outer(sf, actionspace) * (
        1 + np.outer(sf, actionspace) / ab
    )  # produces a sparing factors x actions space array
    return BED


def value_eval(sparing_factors, alpha, beta, bedn=0, abt=10, abn=3, bound=90):
    """
    Calculate the best policy for a list of k sparing factors at the k-1th
    fraction based on a dynamic programming algorithm. Estimation of the
    probability distribution is based on prior patient data.

    sparing_factors: list/array of k sparing factors. A planning sparing factor is necessary!
    abt: alpha beta ratio of tumor
    abn: alpha beta ratio of Organ at risk
    bound: upper limit of BED in OAR'

    return:
        Values: a sparing_factor-2 x BEDT x sf dimensional matrix with the value of each BEDT/sf state
        Values4: Values of the first fraction
        policy: a sparing_factor-2 x BEDT x sf dimensional matrix with the policy of each BEDT/sf state. fourth index = first fraction, first index = last fraction
        policy4: policy of the first fraction

    Note! The sparing factors are not evenly spread from 0-1.4! One should use
    argfind to find the correct index of a desired sparing factor before
    reading the policy!
    """
    mean_sf = np.mean(sparing_factors)
    std = std_calc(sparing_factors, alpha, beta)
    X = get_truncated_normal(mean=mean_sf, sd=std, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf = np.arange(0, 1.31, 0.01)
    sf = sf[prob > 0.00001]
    prob = prob[
        prob > 0.00001
    ]  # we only take values above a certain threshold to lower the computation time
    BEDT = np.arange(0, 90.3, 0.1)
    # we prepare an empty values list and open an action space which is equal
    # to all the dose numbers that can be given in one fraction
    Values = np.zeros(len(BEDT) * len(sf) * 4).reshape(
        4, len(BEDT), len(sf)
    )  # 2d values list with first indice being the BED and second being the sf
    actionspace = np.arange(0, 22.4, 0.1)
    policy = np.zeros((4, len(BEDT), len(sf)))

    upperbound = 90.2
    start = time.time()
    delivered_doses = np.round(BED_calc(sf, abn, actionspace), 1)
    BEDT_rew = BED_calc(
        1, abt, actionspace
    )  # this is the reward for the dose deposited inside the normal tissue.
    BEDT_transformed, meaningless = np.meshgrid(BEDT_rew, np.zeros(len(sf)))
    for state in range(
        0, 5
    ):  # We have five fractionations with 2 special cases 0 and 4
        if (
            state == 4
        ):  # first state with no prior dose delivered so we dont loop through BEDT
            future_bed = delivered_doses
            future_bed[
                future_bed > upperbound
            ] = upperbound  # any dose surpassing the upper bound will be set
                            # to the upper bound which will be penalized strongly
            future_values_prob = (
                Values[state - 1][(future_bed * 10).astype(int)] * prob
            ).sum(
                axis=2
            )  # in this array are all future values multiplied with the
               # probability of getting there. shape = sparing factors x actionspace
            penalties = np.zeros(future_bed.shape)
            penalties[future_bed > bound] = (
                -(future_bed[future_bed > bound] - bound) * 5
            )
            Vs = future_values_prob + BEDT_transformed + penalties

            policy4 = Vs.argmax(axis=1)
            Values4 = Vs.max(axis=1)

        else:
            future_values_prob_all = (Values[state - 1] * prob).sum(axis=1)
            for bed in range(
                len(BEDT)
            ):  # this and the next for loop allow us to loop through all states
                future_bed = delivered_doses + bed / 10
                future_bed[
                    future_bed > upperbound
                ] = upperbound  # any dose surpassing 95 is set to 95.
                if state == 0:  # last state no more further values to add
                    penalties = np.zeros(future_bed.shape)
                    penalties[future_bed > bound] = (
                        -(future_bed[future_bed > bound] - bound) * 5
                    )
                    penalties[
                        future_bed == upperbound
                    ] = (
                        -10000
                    )  # here we produced the penalties for all the values surpassing the limit
                    Vs = (
                        BEDT_transformed + penalties
                    )  # Value of each sparing factor for each action
                else:
                    penalties = np.zeros(future_bed.shape)
                    penalties[future_bed == upperbound] = -100
                    future_values_prob = future_values_prob_all[
                        (future_bed * 10).astype(int)
                    ]  # in this array are all future values multiplied with
                       # the probability of getting there. shape = sparing factors x actionspace
                    Vs = future_values_prob + BEDT_transformed + penalties

                best_action = Vs.argmax(axis=1)
                valer = Vs.max(axis=1)
                policy[state][bed] = best_action
                Values[state][bed] = valer
        print(str(state + 1) + " loop done")
    end = time.time()
    print("time elapsed = " + str(end - start))
    if len(sparing_factors) == 2:
        optaction = actionspace[policy4[argfind(sf, sparing_factors[1])]]
    else:
        if bedn == 0:
            print(
                "Total dose delivered to OAR is missing for optimal action calculation"
            )
            optaction = "unknown"
        else:
            optaction = actionspace[
                policy[6 - len(sparing_factors)][int(round(bedn, 1) * 10)][
                    argfind(sf, sparing_factors[-1])
                ].astype(int)
            ]
    print("optimal dose in fraction", len(sparing_factors) - 1, "= ", optaction)
    return [Values, policy, Values4, policy4, sf]


# this function is only needed as long as the BEDT and the actionspace are
# finally not specified, as it can be solved faster with a bit of algebra
def argfind(
    searched_list, value
):
    index = min(range(len(searched_list)), key=lambda i: abs(searched_list[i] - value))
    return index


# this function calculates the fractionation plan according to the
# reinforcement learning
def result_calc_BEDNT(
    pol4, pol, sf, sparing_factors, abt=10, abn=3
):
    """
    In this function gives the treatment plan for a set of sparing factors
    based on the sparing factors that have been used to calculate the optimal
    policy the pol4 and pol matrices are the ones that are returnedin the
    value_eval function.

    pol4: first fraction policy
    pol: second - fifth fraction policy
    sf: list of sparing factors used in value_eval. This list tells us on what index a specific sparing factor is.
    sparing_factors: sparing factors that should be used to make a plan. list starting from first fraction
    """
    actionspace = np.arange(0, 22.4, 0.1)
    total_bedt = BED_calc0(actionspace[pol4[argfind(sf, sparing_factors[0])]], abt)
    total_bednt = BED_calc0(
        actionspace[pol4[argfind(sf, sparing_factors[0])]], abn, sparing_factors[0]
    )
    print(
        "fraction 1 dose delivered: ",
        actionspace[pol4[argfind(sf, sparing_factors[0])]],
    )
    print(
        "total accumulated  biological effective dose in tumor; fraction 1 = ",
        round(total_bedt, 1),
    )
    print(
        "total accumulated  biological effective dose in normal tissue; fraction 1 = ",
        round(total_bednt, 1),
    )
    for index, fraction in enumerate(range(3, -1, -1)):
        dose_action = actionspace[
            pol[fraction][(round(total_bednt, 1) * 10).astype(int)][
                argfind(sf, sparing_factors[index + 1])
            ].astype(int)
        ]
        dose_delivered = BED_calc0(dose_action, abt)
        total_bedt += dose_delivered
        total_bednt += BED_calc0(dose_action, abn, sparing_factors[index + 1])
        print("fraction ", index + 2, "dose delivered: ", round(dose_action, 1))
        print(
            "total accumulated dose in tumor; fraction ",
            index + 2,
            "=",
            round(total_bedt, 1),
        )
        print(
            "total accumulated dose in normal tissue; fraction ",
            index + 2,
            "=",
            round(total_bednt, 1),
        )
