import numpy as np
import reinforce.plan as plan
import scipy.optimize as opt
import scipy.interpolate as interpol
import matplotlib.pyplot as plt

def fractions(n):
    return np.arange(2, n+1)

def B_static(sf, d):
    # BED^N calculation for 1 therapy
    # with static dose
    b = sf*d * (1+sf*d/params['abn']) 
    return np.sum(b)

def B_aft(algorithm, params):
    # BED^N calculation for 1 therapy
    relay = plan.multiple(algorithm, params)
    return relay[0]

def C_n_linear(n, c):
    # cost from using additional fraction
    lin = fractions(n)
    return c * lin

def B_n(n, param, reps):
    # BED^N calculation for list of fractions
    # and sampled reps times for each fraction
    ab = param['abt']
    goal = param['tumor_goal']
    mu = param['fixed_mean']
    sigma = param['fixed_std']
    cumulative_dose = np.zeros(reps)
    cumulative_dose_static = np.zeros(reps)
    B = np.zeros((2, n-1))
    for i, n in enumerate(fractions(n)):
        for j in range(reps):
            sf_list = np.random.normal(mu,
                sigma, n+1)
            param['number_of_fractions'] = n
            param['sparing_factors'] = sf_list
            cumulative_dose[j] = B_aft('oar', param)
            physical_dose = (np.sqrt(n*ab*(n*ab+4*goal)) - n*ab) / (2*n*mu)
            cumulative_dose_static[j] = B_static(sf_list, physical_dose)
        B[0][i] = np.mean(cumulative_dose)
        B[1][i] = np.mean(cumulative_dose_static)
    return B

def f_n(n):
    # BED^N cost
    B = B_n(n, params, num_samples)
    x = fractions(n)
    return np.array((x, B[0], B[1]))

def F_n_fit(n, D, ab):
    bed = D * (1 + D/(n * ab))
    return bed

def F_total(n, c, D, ab):
    bed = D * (1 + D/(n * ab))
    return bed + c * n

def F_diff(c, n, n_target):
    fn = f_n(n)
    fn_min = np.argmin(fn[1])
    n_guess = fn[0][fn_min]
    popt, _ = opt.curve_fit(F_n_fit, fn[0], fn[1],
                    p0=(params['tumor_goal'], params['abn']))
    n_min = opt.minimize(F_total, n_guess, method='Nelder-Mead',
                    args=(c, popt[0], popt[1])).x[0]
    diff = np.abs(n_target - n_min)
    return diff

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

N = 10
C = 1.3
n_target = 5
num_samples = 13
plot = 'w/o C'


fn = f_n(N) # w/o C
fn_tot = np.copy(fn)
fn_tot[1] += C_n_linear(N, C) # w/ C
fn_tot[2] += C_n_linear(N, C)
x = np.arange(2, N, 0.3)
# curve fits
popt, _ = opt.curve_fit(F_n_fit, fn[0], fn[1],
                    p0=(params['tumor_goal'], params['abn']))
popt2, _ = opt.curve_fit(F_n_fit, fn[0], fn[2],
                    p0=(params['tumor_goal'], params['abn']))
# interpolation
f_n_int = interpol.interp1d(fn[0], fn[1], kind='quadratic')
# minimisation
n_test = opt.minimize(F_total, [1], method='Nelder-Mead', args=(C, popt[0], popt[1])).x[0]
print(f'With C= {C}, n_min = {n_test}')

# c_opti = opt.minimize(F_diff, [1], method='Nelder-Mead', args=(N, n_target)).x[0]
# print(f'With n_target= {n_target}, C = {c_opti}')

if plot == 'w/o C':
    plt.scatter(fn[0], fn[1], label='aft', marker='x')
    plt.scatter(fn[0], fn[2], label='no aft', marker='1')
    plt.plot(x, F_n_fit(x, popt[0], popt[1]), label='fit_aft')
    plt.plot(x, F_n_fit(x, popt2[0], popt2[1]), label='fit_no_aft')
    plt.plot(x, f_n_int(x), label='interpol')

elif plot == 'w/ C':
    plt.scatter(fn[0], fn_tot[1], label='aft', marker='x')
    plt.scatter(fn[0], fn_tot[2], label='no aft', marker='v')
    plt.plot(x, F_total(x, C, popt[0], popt[1]), label='fit_aft')
    plt.plot(x, F_total(x, C, popt2[0], popt2[1]), label='fit_no_aft')

plt.legend()
plt.show()

# F_n(1, 7, 5)