import interpol2D_OARminfrac as intmin

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma, truncnorm

n_frac = 4
[a, b] = intmin.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
sf = np.linspace(0.95, 0.98, n_frac)
c_list = np.linspace(0, 100000, 5)
dose_delivery = np.zeros((len(c_list),n_frac))
for i, c in enumerate(c_list):
    relay = intmin.whole_plan(n_frac, sf, a, b, 30, C=c, max_dose=100)
    relay2 = np.array(relay)[0][:]
    print(relay2)
    #dose_delivery[i] = relay2

#print(dose_delivery)