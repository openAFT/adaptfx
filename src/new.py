import numpy as np
import reinforce.plan as plan
import scipy.optimize as opt
import scipy.interpolate as interpol
import matplotlib.pyplot as plt

def fractions(n):
    return np.arange(2, n+1)

def B_calc(algorithm, params):
    # BED^N calculation for single fraction 
    relay = plan.multiple(algorithm, params)
    return relay[0:2]

def C_n_linear(c, n):
    # cost from using additional fraction
    lin = fractions(n)
    return c * lin

def B_n(n, param, reps):
    # BED^N calculation for list of fractions
    # and sampled reps times for each fraction
    mu = param['fixed_mean']
    sigma = param['fixed_std']
    cumulative_dose = np.zeros(reps)
    B = np.zeros((2, n-1))
    for i, n in enumerate(fractions(n)):
        for j in range(reps):
            sf_list = np.random.normal(mu,
                sigma, n+1)
            param['number_of_fractions'] = n
            param['sparing_factors'] = sf_list
            cumulative_dose[j] = B_calc('oar', param)[0]
        B[0][i] = np.mean(cumulative_dose)
        B[1][i] = np.std(cumulative_dose)
    return B

def f_n(n, c):
    # sum linear cost with BED^N cost
    B = B_n(n, params, number_of_rep)[0]
    C = C_n_linear(c, n)
    return np.array((B + C, fractions(n)))

# def F_n(c, n, n_target):
#     fs = f_n(n, c)
#     min_index = np.argmin(fs[0])
#     n_min = fs[1][min_index]
#     diff = np.abs(n_target - n_min)
#     return diff

def F_n_fit(n, D, ab):
    bed = D * (1 + D/(n * ab))
    return C * n + bed

params = {
            'number_of_fractions': 0,
            'sparing_factors': None,
            'fixed_prob': 1,
            'fixed_mean': 0.9,
            'fixed_std': 0.04,
            'tumor_goal': 30,
            'OAR_limit': 5,
            'C': None,
            'alpha': None,
            'beta': None,
            'max_dose': 22.3,
            'min_dose': 0,
            'abt': 10,
            'abn': 3
            }

N = 7
C = 2
n_target = 5
number_of_rep = 18
f_n_real = f_n(N, C)
x = np.arange(2, N, 0.3)

# popt, pcov = opt.curve_fit(F_n_fit, f_n_real[1], f_n_real[0],
#                     p0=(params['tumor_goal'], params['abn']))
# print('D, ab fitted:',popt)

f_n_int = interpol.interp1d(f_n_real[1], f_n_real[0], kind='linear')

plt.plot(fractions(N), f_n_real[0], label='real')
plt.plot(x, F_n_fit(x, params['tumor_goal'], params['abn']), label='no AFT')
#plt.plot(x, F_n_fit(x, popt[0], popt[1]), label='fit')
plt.plot(x, f_n_int(x), label='interpol')
plt.legend()
plt.show()

# test = opt.minimize(F_n, [0], method='Nelder-Mead', args=(N, N-1))

# F_n(1, 7, 5)