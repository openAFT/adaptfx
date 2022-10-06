import numpy as np
import reinforce.plan as plan
import scipy.optimize as opt
import h5py as hdf
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

def d_T(n, abt, goal):
    dose = (np.sqrt(n*abt*(n*abt+4*goal)) - n*abt) / (2*n)
    return dose

def Bn(n_max, para, n_samples):
    # BED^N calculation for list of fractions
    # and sampled reps times for each fraction
    # returns 
    abt = para['abt']
    goal = para['tumor_goal']
    mu = para['fixed_mean']
    sigma = para['fixed_std']
    BED_aft = np.zeros(n_max-1)
    BED_noaft = np.zeros(n_max-1)
    x = fractions(n_max)
    for i, n_max in enumerate(fractions(n_max)):
        BED_list_aft = np.zeros(n_samples)
        BED_list_noaft = np.zeros(n_samples)
        physical_dose = d_T(n_max, abt, goal)
        para['number_of_fractions'] = n_max
        for j in range(n_samples):
            sf_list = np.random.normal(mu,
                sigma, n_max+1)
            para['sparing_factors'] = sf_list
            BED_list_aft[j] = B_aft('oar', para)
            BED_list_noaft[j] = B_noaft(sf_list, physical_dose, para)
        BED_aft[i] = np.mean(BED_list_aft)
        BED_noaft[i] = np.mean(BED_list_noaft)
    return np.array((x, BED_noaft, BED_aft))

def Cn(n_max, c):
    # cost from using additional fraction
    lin = fractions(n_max)
    return c * lin

class FitClass:
    '''
    This class is used to fit to several functions
    which also enables the use of parameters
    '''
    def __init__(self):
        pass

    def B_func(self, n, sf, c=None):
        para = self.para
        d_tumor = d_T(n ,para['abt'], para['tumor_goal'])
        bed = sf**2 * n * d_tumor * (1/sf - para['abt']/para['abn']) + \
            sf **2 * para['tumor_goal'] * para['abt']/para['abn']
        if c == None:
            return bed
        else:
            return bed + c * n

def Bn_fit(func, x, y):
    popt, _ = opt.curve_fit(func, x, y)
    return popt

def Fn(n_max, c, bed):
    # returns total cost
    C = Cn(n_max, c)
    total_cost = np.copy(bed)
    total_cost[1] += C
    total_cost[2] += C
    return total_cost

def c_find(n_max, n_targ, c_list, bed):
    m = np.shape(c_list)
    n_list_aft = np.zeros(m)
    n_list_noaft = np.zeros(m)
    for j, c in enumerate(c_list):
        fn = Fn(n_max, c, bed)
        i_noaft = np.argmin(fn[1])
        i_aft = np.argmin(fn[2])
        n_list_aft[j] = fn[0][i_aft]
        n_list_noaft[j] = fn[0][i_noaft]
    c_found = np.zeros((3,2))
    c_mask_aft = np.ma.masked_where(n_list_aft==n_targ, c_list).mask
    c_mask_noaft = np.ma.masked_where(n_list_noaft==n_targ, c_list).mask
    c_found[1] = c_list[c_mask_noaft][[0,-1]]
    c_found[2] = c_list[c_mask_aft][[0,-1]]
    return c_found

def c_find_root(n_targ, sf, func):
    def thing(c):
        f_min = opt.minimize(func, n_targ, args=(sf, c)).x
        return n_targ - f_min[0]
    c_opt = opt.root(thing, 1).x
    return c_opt[0]

params = {
            'number_of_fractions': 0,
            'sparing_factors': None,
            'fixed_prob': 1,
            'fixed_mean': 0.9,
            'fixed_std': 0.04,
            'tumor_goal': 72,
            'oar_limit': None,
            'C': None,
            'alpha': None,
            'beta': None,
            'max_dose': 22.3,
            'min_dose': 0,
            'abt': 10,
            'abn': 3
            }

N_max = 12
N_target = 8
C_list = np.arange(1,7,0.05)
num_samples = 1000
filename = 'work/BED_t72_n12_1000.hdf5'
plot = 1
write = 0

if write:
    bn = Bn(N_max, params, num_samples)
    with hdf.File(filename, 'w') as hf:
        hf.create_dataset('bn', data=bn)
else:
    with hdf.File(filename, 'r') as hf:
        bn = hf['bn'][:]

valid_c = c_find(N_max, N_target, C_list, bn)
print(valid_c)

instance = FitClass()
instance.para = params
sf_fit, _ = Bn_fit(instance.B_func, bn[0], bn[2])
sf_fit_no, _ = Bn_fit(instance.B_func, bn[0], bn[1])
x = np.arange(2, N_max, 0.3)
c = valid_c[2][0]
c_no = valid_c[1][0]

c_opt_no = c_find_root(N_target, sf_fit_no, instance.B_func)
print(c_opt_no)
# c_opt = c_find_root(N_target, sf_fit, instance.B_func)
# print(c_opt)

if plot:
    fn1 = Fn(N_max, c_no, bn)
    # fn2 = Fn(N_max, c, bn)
    plt.scatter(bn[0], bn[1], label='no aft', marker='x')
    # plt.scatter(bn[0], bn[2], label='aft', marker='x')
    plt.scatter(fn1[0], fn1[1], label='noaftlow', marker='1')
    # plt.scatter(fn2[0], fn2[1], label='aftlow', marker='1')
    plt.plot(x, instance.B_func(x, sf_fit_no))
    plt.plot(x, instance.B_func(x, sf_fit_no, c))

    plt.ylabel('cost')
    plt.xlabel('fraction $n$')
    plt.legend()
    plt.show()