import numpy as np
import interpol2D_OARminfrac as intmin
import interpol2D_OAR as intoar

typ = 1
n_frac = 5
sf = 1
sf_array = np.ones(n_frac+1)*sf
sf_array[3] = 0.7
mu = np.mean(sf_array)
sigma = np.std(sf_array)
c_list = np.linspace(0, 6, n_frac)
dose_delivery = np.zeros((len(c_list),n_frac))

if typ==1:
    dose_delivery = np.zeros((len(c_list),n_frac))
    for i, c in enumerate(c_list):
        relay = intmin.whole_plan(number_of_fractions=n_frac,
                                  sparing_factors=sf_array,
                                  alpha=None,
                                  beta=None,
                                  goal=30,
                                  max_dose=100,
                                  C=c,
                                  fixed_prob=1,
                                  fixed_mean=mu,
                                  fixed_std=sigma
                                  )
        relay2 = np.array(relay)[2][:]
        dose_delivery[i] = relay2

elif typ==2:
    [a, b] = intmin.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
    sf = np.linspace(1, 1.1, n_frac)
    c_list = [10000]
    dose_delivery = np.zeros((len(c_list),n_frac))
    for i, c in enumerate(c_list):
        relay = intmin.whole_plan(n_frac, sf, a, b, 30, C=c, max_dose=100)
        relay2 = np.array(relay)[0][:]
        dose_delivery[i] = relay2

elif typ==3:
    [a, b] = intoar.data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
    print(a,b)
    sf = np.linspace(0.4, 0.5, n_frac)
    relay = intoar.whole_plan(n_frac, sf, a, b, 30, max_dose=100)
    relay2 = np.array(relay)[0][:]
    print(relay2) 

print(dose_delivery)