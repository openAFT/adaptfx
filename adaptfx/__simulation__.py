import interpol2D_OARminfrac as intmin
import interpol2D_OAR as intoar

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma, truncnorm

typ = 1


if typ:
    n_frac = 8
    [a, b] = intmin.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
    sf = np.linspace(1, 1.1, n_frac)
    c_list = np.linspace(0, 8000, 5)
    dose_delivery = np.zeros((len(c_list),n_frac))
    for i, c in enumerate(c_list):
        relay = intmin.whole_plan(n_frac, sf, a, b, 30, C=c, max_dose=100)
        relay2 = np.array(relay)[0][:]
        print(relay2)
        #dose_delivery[i] = relay2

    #print(dose_delivery)

#if not typ:
    # n_frac = 4
    # [a, b] = intoar.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
    # print(a,b)
    # sf = np.linspace(0.4, 0.5, n_frac)
    # c_list = np.linspace(0, 100000, 5)
    # dose_delivery = np.zeros((len(c_list),n_frac))


    # relay = intoar.whole_plan(n_frac, sf, a, b, 30, max_dose=100)
    # relay2 = np.array(relay)[0][:]
    # print(relay2) 