import adaptfx as afx
import numpy as np
import matplotlib.pyplot as plt

mu, std = 0.9, 0.04
n_frac = 12

c_dict = {
    'number_of_fractions': n_frac,
    'fraction': 0,
    'sparing_factors': None,
    'alpha': 'invalid',
    'beta': 'invalid',
    'abt': 10,
    'abn': 3,
    'fixed_prob': 1,
    'fixed_mean': mu,
    'fixed_std': std,
    'accumulated_tumor_dose': 0,
    'accumulated_oar_dose': 0,
    'min_dose': 0,
    'max_dose': -1,
    'tumor_goal': 72,
    'c': 4.39,
        }

sets = {
    'dose_stepsize': 0.5,
    'state_stepsize': 0.5,
    'sf_stepsize': 0.01,
    'sf_low': 0.7,
    'sf_high': 1.1,
    'sf_prob_threshold': 0,
    'inf_penalty': 1e5
        }

def doses(input_dict, input_sets):
    def sub_multiple(input_dict, input_sets):
        dose_list = afx.multiple('frac', input_dict, input_sets)[0][2][0]
        return np.count_nonzero(~np.isnan(dose_list))

    # multiple = np.vectorize(sub_multiple, otypes=[dict, dict], excluded=['algorithm'])
    physical_dose = sub_multiple(input_dict, input_sets)
    return physical_dose

n_patients = 250
plans = np.zeros(n_patients)
for i in range(n_patients):
    c_dict['sparing_factors'] = list(np.random.normal(mu, std, n_frac + 1))
    plans[i] = doses(c_dict, sets)
plans_hist = np.histogram(plans, bins=np.arange(0.5,n_frac +1,1))
print(plans_hist)

plt.hist(plans, bins=np.arange(0.5,n_frac +1,1))
plt.show()