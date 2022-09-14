import numpy as np
import reinforce.plan as plan
import scipy.optimize as opt
# import scipy.interpolate as interpol
import matplotlib.pyplot as plt

def B_calc(algorithm, params):
    relay = plan.multiple(algorithm, params)
    return relay[0:2]

def fractions(n):
    return np.arange(2, n+1)

def C_n_linear(C, n):
    lin = fractions(n)
    return C * lin

def B_n(n, param, reps):
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
    B = B_n(n, params, number_of_rep)[0]
    C = C_n_linear(c, n)
    return np.array((B + C, fractions(n)))

def F_n(c, n, n_target):
    fs = f_n(n, c)
    min_index = np.argmin(fs[0])
    n_min = fs[1][min_index]
    print(fs)
    print(n_min)
    diff = np.abs(n_target - n_min)
    return diff

def F_n_fit(n, D):
    bed = D * (1 + D/(n * params['abn']))
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

N = 5
C = 2
n_target = 5
number_of_rep = 3

data = f_n(N, C)

popt, pcov = opt.curve_fit(F_n_fit, data[1], data[0], p0=(30))
print(popt)

plt.plot(fractions(N), F_n_fit(N, popt[0]))
plt.show()

# test = opt.minimize(F_n, [0], method='Nelder-Mead', args=(N, N-1))

# F_n(1, 7, 5)