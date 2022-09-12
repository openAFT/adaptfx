import numpy as np
import reinforce.plan as plan
from common.maths import std_calc

def B_calc(algorithm, params):
    relay = plan.multiple(algorithm, params)
    return relay[0:2]

fixed_mean = 0.9
fixed_std = 0.04

params = {
            'number_of_fractions': 0,
            'sparing_factors': None,
            'fixed_prob': 1,
            'fixed_mean': fixed_mean,
            'fixed_std': fixed_std,
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

number_of_fractions = 7
number_of_sf = 2
cumulative_dose = np.zeros((number_of_fractions-2, number_of_sf))

for i, n in enumerate(range(2, number_of_fractions)):
    for j in range(number_of_sf):
        sf_list = np.random.normal(fixed_mean,
            fixed_std, n+1)
        params['number_of_fractions'] = n
        params['sparing_factors'] = sf_list
        cumulative_dose[i][j] = B_calc('oar', params)[0]

print(cumulative_dose)