# -*- coding: utf-8 -*-
"""
whole plan 3D interpolation. This algorithm tracks tumor and OAR BED. If the prescribed tumor dose can be reached, the OAR dose is minimized. If the prescribed tumor dose can not be reached while staying below
maximum BED, the tumor dose is maximized. The value_eval function calculates the optimal dose for one sparing factor given a sparing factor list and the alpha and beta hyperparameter of previous data (can be calculated with data_fit).
the whole_plan function calculates the whole plan given all sparing factors and the hyperparameters
"""

import numpy as np
from scipy.stats import truncnorm
import time
from scipy.stats import invgamma
from scipy.interpolate import RegularGridInterpolator



def data_fit(data):
    '''This function fits the alpha and beta value for the conjugate prior
    input: data: a nxk matrix with n the amount of patints and k the amount of sparing factors per patient'''
    variances = data.var(axis = 1)
    alpha,loc,beta = invgamma.fit(variances, floc = 0)
    return[alpha,beta]


def get_truncated_normal(mean=0, sd=1, low=0.01, upp=10):
    '''produces a truncated normal distribution'''
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def probdist(X):
    '''This function produces a probability distribution based on the normal distribution X '''
    prob = np.zeros(130)
    idx=0
    for i in np.arange(0.01,1.31,0.01):
        prob[idx] = X.cdf(i+0.004999999999999999999)-X.cdf(i-0.005)
        idx +=1
    return prob
    
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

def argfind(searched_list,value): 
    """ This function is used to find the index of certain values.
    searched_list: list/array with values
    value: value that should be inside the list
    return: index of value"""
    index = min(range(len(searched_list)), key=lambda i: abs(searched_list[i]-value))
    return  index
    


def BED_calc0( dose, ab,sparing = 1):
    '''calculates the BED for a specific dose'''
    BED = sparing*dose*(1+(sparing*dose)/ab)
    return BED

def BED_calc_matrix(actionspace,ab,sf):
    '''calculates the BED for an array of values'''
    BED = np.outer(sf,actionspace)*(1+np.outer(sf,actionspace)/ab) #produces a sparing factors x actions space array
    return BED


    
def value_eval(fraction,BED_OAR,BED_tumor,sparing_factors,abt,abn,bound_OAR,bound_tumor,alpha,beta): #there is still a mistake somewhere. Maybe try with only maximizing dose.
    """Calculates the optimal dose for the desired fraction.
    fraction: number of actual fraction (1 for first, 2 for second, etc.)
    BED_OAR: accumulated BED in OAR (from previous fractions) zero in fraction 1
    BED_tumor: accumulated BED in tumor (from previous fractions) zero in fraction 1
    sparing_factors: list or array of all sparing factors that have been observed. e.g. list of 3 sparing factors in fraction 2 (planning,fx1,fx2)
    abt: alpha-beta ratio of tumor
    abn: alpha-beta ratio of OAR
    bound_OAR: maximal BED of OAR
    bound_tumor: prescribed tumor BED
    alpha: alpha hyperparameter of std prior derived from previous patients
    beta: beta hyperparameter of std prior derived from previous patients
    return:
    actual_policy: optimal physical dose for actual fraction
    accumulated_tumor_dose: accumulated tumor BED
    accumulated_OAR_dose: accumulated OAR BED
    """
    mean = np.mean(sparing_factors) #extract the mean and std to setup the sparingfactor distribution
    standard_deviation = std_calc(sparing_factors,alpha,beta)
    X = get_truncated_normal(mean= mean, sd=standard_deviation, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf= np.arange(0.01,1.31,0.01)
    sf = sf[prob>0.00001] #get rid of all probabilities below 10^-5
    prob = prob[prob>0.00001]
    underdosepenalty = 10
    BEDT = np.arange(BED_tumor, bound_tumor,1) #tumordose 
    BEDNT = np.arange(BED_OAR,bound_OAR,1) #OAR dose
    BEDNT = np.concatenate((BEDNT,[bound_OAR,bound_OAR + 1]))
    BEDT = np.concatenate((BEDT,[bound_tumor, bound_tumor + 1]))
    Values = np.zeros([(5-fraction),len(BEDT),len(BEDNT),len(sf)]) #2d values list with first indice being the BED and second being the sf
    actionspace = np.arange(0,22.3,0.1)
    policy = np.zeros(((5-fraction),len(BEDT),len(BEDNT),len(sf)))
    upperbound_normal_tissue = bound_OAR + 1 
    upperbound_tumor = bound_tumor + 1

    OAR_dose = BED_calc_matrix(actionspace,abn,sf) #calculates the dose that is deposited into the normal tissue for all sparing factors            
    tumor_dose = BED_calc_matrix(actionspace,abt,1)[0] #this is the dose delivered to the tumor 
    
    actual_fraction_sf = argfind(sf,np.round(sparing_factors[-1],2))


    
    for index,frac_state_plus in enumerate(np.arange(6,fraction,-1)): #We have five fractionations with 2 special cases 0 and 4
        frac_state = frac_state_plus-1
        if frac_state == 1: #first state with no prior dose delivered so we dont loop through BEDNT
            future_OAR = BED_OAR + OAR_dose[actual_fraction_sf]
            future_tumor = BED_tumor + tumor_dose
            future_OAR[future_OAR > bound_OAR] = upperbound_normal_tissue #any dose surpassing the upper bound will be set to the upper bound which will be penalized strongly
            future_tumor[future_tumor > bound_tumor] = upperbound_tumor
            future_values_prob = (Values[index-1]*prob).sum(axis=2) #future values of tumor and oar state
            value_interpolation = RegularGridInterpolator((BEDT,BEDNT),future_values_prob)
            future_value_actual = value_interpolation(np.array([future_tumor,future_OAR]).T) 
            Vs = future_value_actual - OAR_dose[actual_fraction_sf]
            actual_policy = Vs.argmax(axis=0)            
        
        elif frac_state == fraction: #if we are in the actual fraction we do not need to check all possible BED states but only the one we are in
            if fraction != 5: 
                future_OAR = BED_OAR + OAR_dose[actual_fraction_sf]
                future_OAR[future_OAR > bound_OAR] = upperbound_normal_tissue #any dose surpassing the upper bound will be set to the upper bound which will be penalized strongly
                future_tumor = BED_tumor + tumor_dose
                future_tumor[future_tumor > bound_tumor] = upperbound_tumor
                future_values_prob = (Values[index-1]*prob).sum(axis=2) #future values of tumor and oar state
                value_interpolation = RegularGridInterpolator((BEDT,BEDNT),future_values_prob)
                future_value_actual = value_interpolation(np.array([future_tumor,future_OAR]).T) 
                Vs = future_value_actual - OAR_dose[actual_fraction_sf]
                actual_policy = Vs.argmax(axis=0)
            else:
                penalty = 0
                sf_end = sparing_factors[-1]
                best_action_BED = (-sf_end+np.sqrt(sf_end**2+4*sf_end**2*(bound_OAR-BED_OAR)/abn))/(2*sf_end**2/abn)
                best_action_tumor = (-1+np.sqrt(1+4*1*(bound_tumor-BED_tumor)/abt))/(2*1**2/abt)
                best_action = np.min([best_action_BED,best_action_tumor],axis=0)
                if BED_OAR > bound_OAR or BED_tumor > bound_tumor:
                    best_action = 0
                    penalty = -100000000000
                if best_action <0:
                    best_action = 0
                future_tumor = BED_tumor+BED_calc0(best_action,abt)
                end_penalty = (future_tumor-bound_tumor) #the farther we are away from the prescribed dose, the
                future_OAR = BED_OAR + BED_calc0(best_action,abn,sf_end)
                actual_policy = best_action*10
        else: 
            future_values_prob = (Values[index-1]*prob).sum(axis=2) #future values of tumor and oar state
            value_interpolation = RegularGridInterpolator((BEDT,BEDNT),future_values_prob) #interpolation function
            for tumor_index, tumor_value in enumerate(BEDT):
                for OAR_index, OAR_value in enumerate(BEDNT): #this and the next for loop allow us to loop through all states
                    penalty = 0
                    future_OAR = OAR_dose + OAR_value
                    future_OAR[future_OAR > bound_OAR] = upperbound_normal_tissue #any dose surpassing 90.1 is set to 90.1
                    future_tumor = tumor_value + tumor_dose
                    future_tumor[future_tumor > bound_tumor] = upperbound_tumor #any dose surpassing the tumor bound is set to tumor_bound + 0.1

                   
                    if frac_state == 5: #last state no more further values to add   
                        best_action_BED = (-sf+np.sqrt(sf**2+4*sf**2*(bound_OAR-OAR_value)/abn))/(2*sf**2/abn) #calculate maximal dose that can be delivered to OAR and tumor
                        best_action_tumor = (-np.ones(len(sf))+np.sqrt(np.ones(len(sf))+4*np.ones(len(sf))*(bound_tumor-tumor_value)/abt))/(2*np.ones(len(sf))**2/abt)
                        best_action = np.min([best_action_BED,best_action_tumor],axis=0) #take the smaller of both doses to not surpass the limit                
                        if OAR_value > bound_OAR or tumor_value > bound_tumor: #if the limit is already surpassed we add a penaltsy
                            best_action = np.zeros(best_action.shape)
                            penalty = -100000000000 #-inf doesnt work properly as sometimes instead of -inf, a nan value is assigned which messes up the results
                        best_action[best_action<0] = 0 #if there are negative values we set them to zero
                        future_tumor = tumor_value+BED_calc0(best_action,abt)
                        end_penalty = (future_tumor-bound_tumor).clip(max=0)*underdosepenalty #the farther we are away from the prescribed dose, the higher the penalty
                        future_OAR = OAR_value + BED_calc0(best_action,abn,sf)
                        Values[index][tumor_index][OAR_index] = end_penalty - BED_calc0(best_action,abn,sf) + penalty #we also substract all the dose delivered to the OAR so the algorithm tries to minimize it
                        policy[index][tumor_index][OAR_index] = best_action*10 
                    else: 
                        future_value = np.zeros([len(sf),len(actionspace)])
                        for actual_sf in range(0,len(sf)):
                            future_value[actual_sf] = value_interpolation(np.array([future_tumor,future_OAR[actual_sf]]).T)
                        Vs = future_value -OAR_dose
                        best_action = Vs.argmax(axis=1)
                        valer = Vs.max(axis=1)
                        policy[index][tumor_index][OAR_index] = best_action
                        Values[index][tumor_index][OAR_index] = valer
    accumulated_tumor_dose = BED_calc0(actual_policy/10,abt)+BED_tumor
    accumulated_OAR_dose = BED_calc0(actual_policy/10,abn,sparing_factors[-1]) + BED_OAR
    print('physical dose delivered in fraction ',fraction,'  = ', actual_policy/10)
    print('acuumulated tumor dose = ', accumulated_tumor_dose)
    print('accumulated dose in normal tissue = ', accumulated_OAR_dose)

    return [actual_policy,accumulated_tumor_dose,accumulated_OAR_dose]

def whole_plan(sparing_factors,abt,abn,bound_OAR,bound_tumor,alpha,beta):
    """calculates all doses for a 5 fraction treatment (with 6 known sparing factors)
    sparing_factors: list or array of 6 sparing factors that have been observed.
    abt: alpha-beta ratio of tumor
    abn: alpha-beta ratio of OAR
    bound_OAR: maximal BED of OAR
    bound_tumor: prescribed tumor BED
    alpha: alpha hyperparameter of std prior derived from previous patients
    beta: beta hyperparameter of std prior derived from previous patients
    """
    start = time.time()
    tumor_dose = 0
    OAR_dose = 0
    for looper in range(0,5):
        [actual_policy,accumulated_tumor_dose,accumulated_OAR_dose] = value_eval(looper+1,OAR_dose,tumor_dose,sparing_factors[0:looper+2],abt,abn,bound_OAR,bound_tumor,alpha,beta)
        tumor_dose = accumulated_tumor_dose
        OAR_dose = accumulated_OAR_dose
    end = time.time()
    print('time elapsed = ' +str(end - start))
