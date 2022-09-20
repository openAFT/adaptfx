import numpy as np
import reinforce.plan as plan
import scipy.optimize as opt
import matplotlib.pyplot as plt

def fractions(n_max):
    # return x fraction array
    return np.arange(2, n_max+1)

def B_noaft(sf, d, para):
    # BED^N calculation for 1 therapy
    # with static dose
    sf = sf[1:]
    b = sf*d * (1+sf*d/para['abn'])
    return np.sum(b)

def B_aft(algorithm, para):
    # BED^N calculation for 1 therapy
    relay = plan.multiple(algorithm, para)
    return relay[0]

def Bn(n, para, n_samples):
    # BED^N calculation for list of fractions
    # and sampled reps times for each fraction
    # returns 
    ab = para['abt']
    goal = para['tumor_goal']
    mu = para['fixed_mean']
    sigma = para['fixed_std']
    BED_aft = np.zeros(n-1)
    BED_noaft = np.zeros(n-1)
    x = fractions(n)
    for i, n in enumerate(fractions(n)):
        BED_list_aft = np.zeros(n_samples)
        BED_list_noaft = np.zeros(n_samples)
        physical_dose = (np.sqrt(n*ab*(n*ab+4*goal)) - n*ab) / (2*n)
        para['number_of_fractions'] = n
        for j in range(n_samples):
            sf_list = np.random.normal(mu,
                sigma, n+1)
            para['sparing_factors'] = sf_list
            BED_list_aft[j] = B_aft('oar', para)
            BED_list_noaft[j] = B_noaft(sf_list, physical_dose, para)
        BED_aft[i] = np.mean(BED_list_aft)
        BED_noaft[i] = np.mean(BED_list_noaft)
    return np.array((x, BED_noaft, BED_aft))

def Cn(n, c):
    # cost from using additional fraction
    lin = fractions(n)
    return c * lin

def wrong(n, D, ab, c=None):
    bed = D * (1 + D/(n * ab))
    if c == None:
        return bed
    else:
        return bed + c * n

def Bn_fit(x, y, func, para):
    popt, _ = opt.curve_fit(func, x, y, p0=(para['tumor_goal'], para['abn']))
    return popt

def Fn(n, c, bed):
    # returns total cost
    C = Cn(n, c)
    total_cost = np.copy(bed)
    total_cost[1] += C
    total_cost[2] += C
    return total_cost

params = {
            'number_of_fractions': 0,
            'sparing_factors': None,
            'fixed_prob': 1,
            'fixed_mean': 0.9,
            'fixed_std': 0.04,
            'tumor_goal': 72,
            'OAR_limit': None,
            'C': None,
            'alpha': None,
            'beta': None,
            'max_dose': 22.3,
            'min_dose': 0,
            'abt': 10,
            'abn': 3
            }

N_max = 12
C = 1.3
n_target = 5
num_samples = 2

bn = Bn(N_max, params, num_samples)
D_opt, ab_opt = Bn_fit(bn[0], bn[1], wrong, params)
x = np.arange(2, N_max, 0.3)
bn_fit = wrong(x, D_opt, ab_opt)

plt.scatter(bn[0], bn[1])
plt.scatter(bn[0], bn[2])
plt.plot(x, bn_fit)

fn = Fn(N_max, C, bn)
plt.show()


#print(Bn_fit(bn[0], bn[1], params))