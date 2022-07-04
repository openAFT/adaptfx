import interpol2D_OARminfrac as intmin

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma, truncnorm

n_frac = 3
[a, b] = intmin.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
c_list = np.arange(1, 10, 1)**5
dose_delivery = np.zeros((len(c_list),n_frac))
for i, c in enumerate(c_list):
    print(c)
    relay = intmin.whole_plan(n_frac, [0.99, 0.95, 0.58, 0.96], a, b, 50, c)
    relay2 = np.array(relay).transpose()[0][:]
    dose_delivery[i] = relay2

print(dose_delivery)