# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:33:59 2021

@author: yoelh

code without penalties per step
"""



import numpy as np
from scipy.stats import truncnorm
import time
import matplotlib.pyplot as plt



def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
def probdist(X):
    prob = np.zeros(131)
    idx=0
    for i in np.arange(0,1.31,0.01):
        prob[idx] = X.cdf(i+0.004999999999999999999)-X.cdf(i-0.005)
        idx +=1
    return prob

    


def BED_calc( dose, ab,sparing = 1): #this one is bedcalc0
    BED = sparing*dose*(1+(sparing*dose)/ab)
    return BED


def BED_calc_matrix(actionspace, ab, sf): #careful this one has been changed in order
    BED = np.outer(sf,actionspace)*(1+np.outer(sf,actionspace)/ab) #produces a sparing factors x actions space array
    return BED


    
def value_eval(mean,standard_deviation,abt = 10,abn = 3,prescribed_dose = 72):
    X = get_truncated_normal(mean= mean, sd=standard_deviation, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf= np.arange(0,1.31,0.01)
    sf = sf[prob>0.00001] #get rid of all probabilities below 10^-5
    prob = prob[prob>0.00001]
    underdosepenalty = 4
    BEDT = np.arange(0,72.2,0.1) #open up the possible biological effective dose space for tumor
    BEDNT = np.arange(0,90.2,0.1)
    Values = np.zeros(len(BEDT)*len(sf)*4*len(BEDNT)).reshape(4,len(BEDT) ,len(BEDNT),len(sf))
    policy = np.zeros(len(BEDT)*len(sf)*4*len(BEDNT)).reshape(4,len(BEDT) ,len(BEDNT),len(sf))
    actionspace = np.arange(0,22.4,0.1)
    start = time.time()
    upperbound_normal_tissue = 90.1
    upperbound_tumor = 72.1
    start = time.time()
    normal_tissue_dose = np.round(BED_calc_matrix(actionspace,abn,sf),1) #calculates the dose that is deposited into the normal tissue for all sparing factors           
    tumor_dose = BED_calc(actionspace,abt,1) #this is the dose delivered to the tumor
    penalties_tumor = np.zeros(tumor_dose.shape)
    for i,fraction in enumerate(range(5,0,-1)): #We have five fractionations with 2 special cases fraction 1 and 5
        if fraction != 5: #in the last fraction we do not look for the next fraction therefor this code is not needed.
            future_values_prob_all = (Values[fraction-1]*prob).sum(axis = 2) #we can produce all future values multiplied with the probability which gives the future value for each action (independent of the future sf!)
        if fraction == 1: #first fraction with no prior dose delivered so we dont loop through BEDT and BEDNT
            future_bed_tumor = tumor_dose
            future_bed_tumor[future_bed_tumor > upperbound_tumor] = upperbound_tumor #any dose surpassig the upper bound (which is one step above the goal) is set to the threshold of the upper bound which will be penalized strongly so the program NEVER overdoses
            future_bed_normal_tissue = normal_tissue_dose
            future_bed_normal_tissue[future_bed_normal_tissue > upperbound_normal_tissue] = upperbound_normal_tissue #same here with the upper bound
            
            future_values_prob = future_values_prob_all[(future_bed_tumor*10).astype(int)] # extract the needed future valuesmultiplied with their respective probability
            Vs = -normal_tissue_dose.T #this is the penalty for delivering dose to the normal tissue.           
            for index in range(0,len(future_values_prob)):
                Vs[index] += future_values_prob[index][(future_bed_normal_tissue.T[index]*10).astype(int)] #we go through every reached tumor bed and check the reached normal tissue bed for each sparing factor
            
            policy1 = Vs.T.argmax(axis=1)
            Values1 = Vs.T.max(axis=1)
        else:
            for bed in range(len(BEDT)): #we loop not through all tumor dose and normal tissue dose states
                print(bed)
                for bedn in range(len(BEDNT)):
                    penalties_normal_tissue = np.zeros(normal_tissue_dose.shape)
                    future_bed_tumor = tumor_dose + bed/10
                    future_bed_tumor[future_bed_tumor > upperbound_tumor] = upperbound_tumor
                    future_bed_normal_tissue = bedn/10 + normal_tissue_dose
                    future_bed_normal_tissue[future_bed_normal_tissue > upperbound_normal_tissue] = upperbound_normal_tissue
                    Vs = -normal_tissue_dose.T #we need to substract the dose deposited into the normal tissue as additional penalty
                    if fraction == 5: #the last fraction does not require future values, but we need to penalize the difference to the desired goals
                        penalties_tumor[future_bed_tumor == upperbound_tumor] = -200
                        meaningless, penalties_tumor_reshaped= np.meshgrid(np.zeros(len(sf)),penalties_tumor)
                        penalties_normal_tissue[future_bed_normal_tissue == upperbound_normal_tissue] = -200
                        end_penalty = (future_bed_tumor-prescribed_dose).clip(max = 0)*underdosepenalty
                        meaningless, end_penalty_reshaped = np.meshgrid(np.zeros(len(sf)),end_penalty)
                        Vs +=  end_penalty_reshaped + penalties_normal_tissue.T
                    else:
                        future_values_prob = future_values_prob_all[(future_bed_tumor*10).astype(int)]
                        for index in range(0,len(future_values_prob)):
                            Vs[index] += future_values_prob[index][(future_bed_normal_tissue.T[index]*10).astype(int)]
                        
                    best_action = Vs.argmax(axis=0)
                    valer = Vs.max(axis=0)
                    policy[fraction-2][bed][bedn] = best_action
                    Values[fraction-2][bed][bedn] = valer
        print(str(i+1) +' loop done')
    end = time.time()
    print('time elapsed = ' +str(end - start))
    return [Values,policy,Values1,policy1]
                        
                        
if __name__ == "__main__":
    [Val,pol,Val1,pol1] = value_eval(0.8,0.1)

    X = get_truncated_normal(mean= 0.8, sd=0.1, low=0, upp=1.3)
    prob = np.array(probdist(X))
    sf= np.arange(0,1.31,0.01)
    sf = sf[prob>0.00001] #get rid of all probabilities below 10^-5
    prob = prob[prob>0.00001]
                        
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sf,Val1)
    ax1.set_title('Values of first fraction' );
    plt.xlabel('sparing factors')
    plt.ylabel('Value')
    
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sf,pol1)
    ax2.set_title('policy of first fraction');
    plt.xlabel('sparing factors')
    plt.ylabel('Dose delivered in Gy')
    plt.show()
    
def argfind(searched_list,value): #this function is only needed as long as the BEDT and the actionspace are finally not specified, as it can be solved faster with a bit of algebra
    index = min(range(len(searched_list)), key=lambda i: abs(searched_list[i]-value))
    return  index

def result_calc(fraction,totalt,totaln,spar,pol,abt = 10,abn = 3):
    action = pol[fraction-2][round(totalt*10)][round(totaln*10)][argfind(sf,spar)]
    bedt = totalt+BED_calc(action/10,abt)
    bednt = totaln + BED_calc(action/10,abn,spar)
    return[bedt,bednt,action/10]

def plan_calc(sparing_factors, sf, pol1, pol, abt = 10, abn = 3):
    first_action = argfind(sf,sparing_factors[0])
    first_dose = pol1[first_action]
    total_bedt = np.round(BED_calc(first_dose/10,abt),1)
    total_bednt = np.round(BED_calc(first_dose/10,abn,sparing_factors[0]),1)
    print('fraction ', 1, 'dose delivered: ', round(first_dose/10,1))
    print('total accumulated dose in tumor; fraction ', 1, '=', round(total_bedt,1))
    print('total accumulated dose in normal tissue; fraction ', 1, '=', round(total_bednt,1))
    for index in range(0,4):
        dose_action = pol[index][round(total_bedt*10)][round(total_bednt*10)][argfind(sf,sparing_factors[index+1])]/10
        dose_delivered = BED_calc(dose_action,abt)
        total_bedt += dose_delivered
        total_bednt += BED_calc(dose_action,abn,sparing_factors[index+1])
        print('fraction ', index+2, 'dose delivered: ', round(dose_action,1))
        print('total accumulated dose in tumor; fraction ', index+2, '=', round(total_bedt,1))
        print('total accumulated dose in normal tissue; fraction ', index+2, '=', round(total_bednt,1))
    