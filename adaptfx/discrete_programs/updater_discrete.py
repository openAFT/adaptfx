# -*- coding: utf-8 -*-
"""
In this file are all the needed functions to calculate an adaptive fractionation treatment plan. The value_eval and the result_calc function are the only ones that should be used
This file requires all sparing factors to be known, therefore, it isnt suited to do active treatment planning but to analyze patient data.
value_eval and result_calc_BEDNT are the most essential codes. The results from value_eval can be used to calculate a treatment plan with result_calc_BEDNT.
The optimal policies for each fraction can be extracted manually(pol4 = first fraction, first index in pol is the last fraction and the last index is the first fraction). but one must know what index represents which sparing factor
Note: This file does not assume all sparing factors to be known at the start, but simulates the treatment planning as if we would get a new sparing factor at each fraction!
This program uses a discrete state space and does not interpolate between states. Therefore, it is less precise than the interpolation programs
"""

import time

import numpy as np
from scipy.stats import invgamma, truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    '''produces a truncated normal distribution'''
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def std_calc(measured_data,alpha,beta):
    '''calculates the most likely standard deviation for a list of k sparing factors and an inverse-gamma conjugate prior
    measured_data: list/array with k sparing factors
    alpha: shape of inverse-gamma distribution
    beta: scale of inverse-gamme distrinbution
    return: most likely std based on the measured data and inverse-gamma prior'''
    n = len(measured_data)
    var_values = np.arange(0.00001,0.25,0.00001)
    likelihood_values = np.zeros(len(var_values))
    for index,value in enumerate(var_values):
        likelihood_values[index] = value**(-alpha-1)/value**(n/2)*np.exp(-beta/value)*np.exp(-np.var(measured_data)*n/(2*value))
    std = (np.sqrt(var_values[np.argmax(likelihood_values)]))
    return std

def distribution_update(sparing_factors, alpha, beta):
    '''produces the updated probability distribution for each fraction based on Variance prior
    sparing_factors: list/array of k spraring factors
    alpha: shape of inverse-gamma distribution
    beta: scale of inverse-gamme distrinbution
    return: k-1 dimensional mean and std arrays starting from the second sparing factor (index 1)
   '''
    means = np.zeros(len(sparing_factors))
    stds = np.zeros(len(sparing_factors))
    for i in range(len(sparing_factors)):
        means[i] = np.mean(sparing_factors[:(i+1)])
        stds[i] = std_calc(sparing_factors[:(i+1)],alpha,beta)
    means = np.delete(means,0)
    stds = np.delete(stds,0) #we get rid of the first value as it is only the planning value and not used in a fraction
    return [means,stds]


def updated_distribution_calc(data,sparing_factors):
    '''calculates the updated distribution based on prior data that is used to setup an inverse gamma distribution
       data shape: nxk where n is the amount of patients and k the amount of sparingfactors per patient
       sparing_factors shape: list/array with k entries with the first sparing factor being the planning sparing factor, therefore not being included in the treatment
       return: updated means and stds for k-1 fractions.'''
    variances = data.var(axis = 1)
    alpha,loc,beta = invgamma.fit(variances, floc = 0) #here beta is the scale parameter
    [means,stds] = distribution_update(sparing_factors,alpha,beta)
    return[means,stds]

def probdistributions(means,stds):
    '''produces the truncated normal distribution for several means and standard deviations
    means: list/array of n means
    stds: list/array of n standard deviations
    return: n probability distributions for values [0.01,1.40]'''
    distributions = np.zeros(141*len(means)).reshape(len(means),141)
    for i in range(len(means)):
        X = get_truncated_normal(means[i], stds[i], low=0, upp=1.4)
        for index,value in enumerate(np.arange(0,1.41,0.01)):
            distributions[i][index] = X.cdf(value+0.004999999999999999999)-X.cdf(value-0.005)
    return distributions

def BED_calc0( dose, ab,sparing = 1):
    BED = sparing*dose*(1+(sparing*dose)/ab)
    return BED

def BED_calc( sf, ab,actionspace):
    BED = np.outer(sf,actionspace)*(1+np.outer(sf,actionspace)/ab) #produces a sparing factors x actions space array
    return BED

def value_eval(sparing_factors,data,abt = 10,abn = 3,bound = 90,riskfactor = 0):
    '''calculates  the best policy for a list of k sparing factors with k-1 fractions based on a dynamic programming algorithm. Estimation of the probability distribution is based on prior patient data
    sparing_factors: list/array of k sparing factors. A planning sparing factor is necessary!
    data: nxk dimensional data of n prior patients with k sparing factors.
    abt: alpha beta ratio of tumor
    abn: alpha beta ratio of Organ at risk
    bound: upper limit of BED in OAR
    riskfactor: "risk reducing" factor of zero is a full adaptive fractionation algorithm  while a sparing factor of 0.1 slightly forces the algorithm to stay close to the 6Gy per fraction plan. a risk factor of 1 results in a 6Gy per fraction plan.
    return:
        Values: a sparing_factor-2 x BEDT x sf dimensional matrix with the value of each BEDT/sf state
        Values4: Values of the first fraction
        policy: a sparing_factor-2 x BEDT x sf dimensional matrix with the policy of each BEDT/sf state. fourth index = first fraction, first index = last fraction
        policy4: policy of the first fraction'''
    sf= np.arange(0,1.41,0.01) #list of all possible sparing factors
    BEDT = np.arange(0,90.3,0.1) #list of all possible Biological effective doses
    Values = np.zeros(len(BEDT)*len(sf)*4).reshape(4,len(BEDT),len(sf)) #2d values list with first indice being the BED and second being the sf
    actionspace = np.arange(0,22.4,0.1) #list of all possible dose actions
    [means,stds] =updated_distribution_calc(data,sparing_factors)
    distributions = probdistributions(means,stds)
    policy = np.zeros((4,len(BEDT),len(sf)))
    upperbound = 90.2
    start = time.time()

    #here we add the calculation of the distance to the standard treatment
    useless,calculator = np.meshgrid(np.zeros(len(actionspace)),sf) #calculator is matrix that has the correct sparing factors
    actionspace_expand,useless = np.meshgrid(actionspace,sf)
    risk_penalty = abs(6/calculator-actionspace_expand)
    delivered_doses = np.round(BED_calc(sf,abn,actionspace),1)
    BEDT_rew = BED_calc(1, abt,actionspace) #this is the reward for the dose deposited inside the normal tissue.
    BEDT_transformed, meaningless = np.meshgrid(BEDT_rew,np.zeros(len(sf)))
    risk_penalty[0] = risk_penalty[1]
    for update_loop in range (0,5):
        prob = distributions[update_loop]
        for state in range(0,5-update_loop): #We have five fractionations with 2 special cases 0 and 4
            print(str(state+1) +' loop done')
            if state == 4: #first state with no prior dose delivered so we dont loop through BEDT
                future_bed = delivered_doses
                future_bed[future_bed > upperbound] = upperbound #any dose surpassing 95 is set to 95. Furthermore, 95 will be penalized so strong that the program avoids it at all costs. (95 is basically the upper bound and can be adapted)
                future_values_prob = (Values[state-1][(future_bed*10).astype(int)]*prob).sum(axis = 2) #in this array are all future values multiplied with the probability of getting there. shape = sparing factors x actionspace
                penalties = np.zeros(future_bed.shape)
                penalties[future_bed > bound] = -(future_bed[future_bed > bound]-bound)*5
                Vs = future_values_prob + BEDT_transformed + penalties - risk_penalty*riskfactor

                policy4 = Vs.argmax(axis=1)
                Values4 = Vs.max(axis=1)

            else:
                future_values_prob_all = (Values[state-1]*prob).sum(axis = 1)
                for bed in range(len(BEDT)): #this and the next for loop allow us to loop through all states
                    future_bed = delivered_doses + bed/10
                    future_bed[future_bed > upperbound] = upperbound #any dose surpassing 95 is set to 95.
                    if state == 0: #last state no more further values to add
                        penalties = np.zeros(future_bed.shape)
                        penalties[future_bed > bound] = -(future_bed[future_bed > bound]-bound)*5
                        penalties[future_bed == upperbound] = -10000  #here we produced the penalties for all the values surpassing the limit
                        Vs = BEDT_transformed + penalties# Value of each sparing factor for each action
                    else:
                        penalties = np.zeros(future_bed.shape)
                        penalties[future_bed == upperbound] = -100
                        future_values_prob = (future_values_prob_all[(future_bed*10).astype(int)])#in this array are all future values multiplied with the probability of getting there. shape = sparing factors x actionspace
                        Vs = future_values_prob + BEDT_transformed + penalties - risk_penalty*riskfactor


                    best_action = Vs.argmax(axis=1)
                    valer = Vs.max(axis=1)
                    policy[state][bed] = best_action
                    Values[state][bed] = valer
    end = time.time()
    print('time elapsed = ' +str(end - start))
    return [Values,policy,Values4,policy4]

def result_calc_BEDNT(pol4,pol,sparing_factors,abt = 10,abn = 3): #this function calculates the fractionation plan according to the reinforcement learning
    '''in this function gives the treatment plan for a set of sparing factors based on the sparing factors that have been used to calculate the optimal policy
the pol4 and pol matrices are the ones that are returnedin the value_eval function
    pol4: first fraction policy
    pol: second - fifth fraction policy
    sparing_factors: sparing factors that should be used to make a plan. list starting from first fraction'''
    actionspace = np.arange(0,22.4,0.1) #list of all possible dose actions
    total_bedt = BED_calc0(actionspace[pol4[round(sparing_factors[0]*100)]],abt)
    total_bednt = BED_calc0(actionspace[pol4[round(sparing_factors[0]*100)]],abn,sparing_factors[0])
    print('fraction 1 dose delivered: ',actionspace[pol4[round(sparing_factors[0]*100)]])
    print('total accumulated  biological effective dose in tumor; fraction 1 = ',round(total_bedt,1))
    print('total accumulated  biological effective dose in normal tissue; fraction 1 = ',round(total_bednt,1))
    for index,fraction in enumerate(range(3,-1,-1)):
        if fraction == 0:
            dose_action = (-sparing_factors[index+1]+np.sqrt(sparing_factors[index+1]**2+4*sparing_factors[index+1]**2*(90-total_bednt)/abn))/(2*sparing_factors[index+1]**2/abn)
        else:
            dose_action = actionspace[pol[fraction][(round(total_bednt,1)*10).astype(int)][round(sparing_factors[index+1]*100)].astype(int)]
        dose_delivered = BED_calc0(dose_action,abt)
        total_bedt += dose_delivered
        total_bednt += BED_calc0(dose_action,abn,sparing_factors[index+1])
        print('fraction ', index+2, 'dose delivered: ', round(dose_action,1))
        print('total accumulated dose in tumor; fraction ', index+2, '=', round(total_bedt,1))
        print('total accumulated dose in normal tissue; fraction ', index+2, '=', round(total_bednt,1))
