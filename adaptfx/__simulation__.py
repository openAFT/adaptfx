import interpol2D_OARminfrac as intmin

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma, truncnorm

n_frac = 4
[a, b] = intmin.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
c_list = np.arange(4.8, 4.92, 0.02)
dose_delivery = np.zeros((len(c_list),n_frac))
for i, c in enumerate(c_list):
    relay = intmin.whole_plan(n_frac, [0.8] * n_frac, a, b, 30, C=c, max_dose=100)
    relay2 = np.array(relay)[0][:]
    dose_delivery[i] = relay2

print(dose_delivery)