# -*- coding: utf-8 -*-
"""
In this file are all functions to calculate the optimal dose for a single fraction, given that all previous sparing factors are known and prior data of patients from the same population is available.
The value_eval function gives the optimal dose for a certain fraction. As input the sparing factors are needed and the alpha and beta hyperparameter of a inverse-gamma distribution to improve the probability distribution.
if the alpha and beta value are not known, the data_fit function can be used which needs the sparing factors of prior patients as input.
The optimal policies can be extracted from pol4 and pol manually (pol4 = first fraction, first index in pol is the last fraction and the last index is the first fraction). But one must know which sparing factor is on which index. To do so one must use the extracted sf from value_eval which tells us which sparing factors have been used on which index.
it is recommended to usethe result_calc_BEDNT to calculate plans with different sparing factors.
In this program an interpolation is used to compute the exact values for each dose. Therefore, it is more precise than the discrete program.
"""

import numpy as np
from scipy.stats import truncnorm
import time
import scipy as sc
from scipy.stats import invgamma







def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    '''produces a truncated normal distribution'''
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def probdist(X):
    '''This function produces a probability distribution based on the normal distribution X '''
    prob = np.zeros(131)
    idx=0
    for i in np.arange(0,1.31,0.01):
        prob[idx] = X.cdf(i+0.004999999999999999999)-X.cdf(i-0.005)
        idx +=1
    return prob
def data_fit(data):
    '''This function fits the alpha and beta value for the conjugate prior
    input: data: a nxk matrix with n the amount of patints and k the amount of sparing factors per patient'''
    variances = data.var(axis = 1)
    alpha,loc,beta = invgamma.fit(variances, floc = 0)
    return[alpha,beta]

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

def BED_calc0( dose, ab,sparing = 1):
    '''calculates the BED for a specific dose'''
    BED = sparing*dose*(1+(sparing*dose)/ab)
    return BED

def BED_calc( sf, ab,actionspace):
    '''calculates the BED for an array of values'''
    BED = np.outer(sf,actionspace)*(1+np.outer(sf,actionspace)/ab) #produces a sparing factors x actions space array
    return BED





def argfind(searched_list,value): #this function is only needed as long as the BEDT and the actionspace are finally not specified, as it can be solved faster with a bit of algebra
    index = min(range(len(searched_list)), key=lambda i: abs(searched_list[i]-value))
    return  index
    

    
def value_eval(fraction,BED,sparing_factors,alpha,beta,abt,abn,bound):
    ''' calculates  the best policy for a list of k sparing factors at the k-1th fraction based on a dynamic programming algorithm. Estimation of the probability distribution is based on prior patient data
    fraction: Actual fraction number (first fraction = 1)
    BED: Dose delivered to OAR so far (calculated in BED)
    sparing_factors: list/array of k sparing factors. A planning sparing factor is necessary! 
    alpha = alpha value of std prior
    beta = beta value of std prior
    abt: alpha beta ratio of tumor
    abn: alpha beta ratio of Organ at risk
    bound: upper limit of BED in OAR'
    return:
        Values: a sparing_factor-2 x BEDT x sf dimensional matrix with the value of each BEDT/sf state
        Values4: Values of the first fraction
        policy: a sparing_factor-2 x BEDT x sf dimensional matrix with the policy of each BEDT/sf state. fourth index = first fraction, first index = last fraction
        policy4: policy of the first fraction
    Note! The sparing factors are not evenly spread from 0-1.4! One should use argfind to find the correct index of a desired sparing factor before reading the policy!'''
    actual_sparing = sparing_factors[-1]
    mean_sf = np.mean(sparing_factors)
    std = std_calc(sparing_factors,alpha,beta)
    X = get_truncated_normal(mean=mean_sf, sd=std, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf= np.arange(0,1.31,0.01)
    sf = sf[prob>0.00001]
    prob = prob[prob>0.00001] #we only take values above a certain threshold to lower the computation time
    BEDT = np.arange(0,90.3,0.1)
    #we prepare an empty values list and open an action space which is equal to all the dose numbers that can be given in one fraction 
    Values = np.zeros(len(BEDT)*len(sf)*4).reshape(4,len(BEDT),len(sf)) #2d values list with first indice being the BED and second being the sf
    actionspace = np.arange(0,22.4,0.1)
    policy = np.zeros((4,len(BEDT),len(sf)))
    
    
    
    
    BEDT = BEDT = np.arange(BED,91.6,1)
    Values = np.zeros(len(BEDT)*len(sf)*(5-fraction)).reshape((5-fraction),len(BEDT),len(sf)) #2d values list with first indice being the BED and second being the sf
    actionspace = np.arange(0,22.4,0.1)
    policy = np.zeros(((5-fraction),len(BEDT),len(sf)))

    
    upperbound = 91
    start = time.time()
    delivered_doses = BED_calc(sf,abn,actionspace)            
    BEDT_rew = BED_calc(1, abt,actionspace) #this is the reward for the dose deposited inside the tumor. 
    BEDT_transformed, meaningless = np.meshgrid(BEDT_rew,np.zeros(len(sf)))
    
    for index,frac_state in enumerate(np.arange(fraction,6)): #We have five fractionations with 2 special cases 0 and 4
        if index == 4: #first state with no prior dose delivered so we dont loop through BEDT
            future_bed = BED + delivered_doses
            future_bed[future_bed > bound] = upperbound #any dose surpassing the upper bound will be set to the upper bound which will be penalized strongly
            value_interpolation = sc.interpolate.interp2d(sf,BEDT,Values[index-1])
            future_value = np.zeros(len(sf)*len(actionspace)*len(sf)).reshape(len(sf),len(actionspace),len(sf))
            for actual_sf in range(0,len(sf)):
                future_value[actual_sf] = value_interpolation(sf,future_bed[actual_sf])
            future_values_prob = (future_value*prob).sum(axis = 2) #in this array are all future values multiplied with the probability of getting there. shape = sparing factors x actionspace
            penalties = np.zeros(future_bed.shape)
            penalties[future_bed > bound] = -1000 #penalizing in each fraction is needed. If not, once the algorithm reached the upper bound, it would just deliver maximum dose over and over again
            Vs = future_values_prob + BEDT_transformed + penalties
            
            actual_policy = Vs.argmax(axis=1)
            actual_value = Vs.max(axis=1)
        
        else:
            if index == 5-fraction: #if we are in the actual fraction we do not need to check all possible BED states but only the one we are in
                if fraction != 5:
                    future_bed = BED + delivered_doses
                    future_bed[future_bed > bound] = upperbound #any dose surpassing the upper bound will be set to the upper bound which will be penalized strongly
                    value_interpolation = sc.interpolate.interp2d(sf,BEDT,Values[index-1])
                    future_value = np.zeros(len(sf)*len(actionspace)*len(sf)).reshape(len(sf),len(actionspace),len(sf))
                    for actual_sf in range(0,len(sf)):
                        future_value[actual_sf] = value_interpolation(sf,future_bed[actual_sf])
                    future_values_prob = (future_value*prob).sum(axis = 2) #in this array are all future values multiplied with the probability of getting there. shape = sparing factors x actionspace
                    penalties = np.zeros(future_bed.shape)
                    penalties[future_bed > bound] = -1000 #penalizing in each fraction is needed. If not, once the algorithm reached the upper bound, it would just deliver maximum dose over and over again
                    Vs = future_values_prob + BEDT_transformed + penalties
                    actual_policy = Vs.argmax(axis=1)
                    actual_value = Vs.max(axis=1)  
                else:
                    best_action = (-sf+np.sqrt(sf**2+4*sf**2*(90-BED)/abn))/(2*sf**2/abn)
                    actual_policy = best_action*10
                    actual_value = BED_calc0(best_action,abt)
            else:                    
                for bed_index, bed_value in enumerate(BEDT): #this and the next for loop allow us to loop through all states
                    future_bed = delivered_doses + bed_value
                    future_bed[future_bed > bound] = upperbound #any dose surpassing 90.1 is set to 90.1
                    if index == 0: #last state no more further values to add                    
                        best_action = (-sf+np.sqrt(sf**2+4*sf**2*(90-bed_value)/abn))/(2*sf**2/abn)
                        if bed_value > 90:
                            best_action = np.zeros(best_action.shape)
                        Values[index][bed_index] = BED_calc0(best_action,abt)                        
                        best_action[best_action > 22.3] = 22.3
                        best_action[best_action<0] = 0
                        policy[index][bed_index] = best_action*10 #this one can be pulled before values. in fact we dont want to deliver unlimited amounts of dose
                    else:
                        penalties = np.zeros(future_bed.shape)
                        penalties[future_bed == upperbound] = -1000 
                        value_interpolation = sc.interpolate.interp2d(sf,BEDT,Values[index-1])
                        future_value = np.zeros(len(sf)*224*(len(sf))).reshape(len(sf),224,len(sf))
                        for actual_sf in range(0,len(sf)):
                            future_value[actual_sf] = value_interpolation(sf,future_bed[actual_sf])
                        future_values_prob = (future_value*prob).sum(axis = 2)                        
                        Vs = future_values_prob + BEDT_transformed + penalties
                        best_action = Vs.argmax(axis=1)
                        valer = Vs.max(axis=1)
                        policy[index][bed_index] = best_action
                        Values[index][bed_index] = valer
    end = time.time()
    print('time elapsed = ' +str(end - start))
    index_sf = argfind(sf,actual_sparing)
    dose_delivered_tumor = BED_calc0(actual_policy[index_sf]/10,abt)
    dose_delivered_OAR = BED_calc0(actual_policy[index_sf]/10,abn,actual_sparing) + BED
    print('physical dose delivered  = ', actual_policy[index_sf]/10)
    print('tumor dose = ', dose_delivered_tumor)
    print('accumulated dose in normal tissue = ', dose_delivered_OAR)




       