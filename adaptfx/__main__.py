import numpy as np
from reinforce import fraction_minimisation as frac
from reinforce import oar_minimisation as oar
from reinforce import tumor_maximisation as tumor
from common.maths import data_fit

def simulation(n_frac):
    [a, b] = data_fit(np.array([[0.99, 0.95, 0.98], [0.95, 0.9, 0.8]]))
    sf = np.linspace(1, 1.1, n_frac)
    c_list = np.linspace(0, 6, 6)
    dose_delivery = np.zeros((len(c_list),n_frac))
    for i, c in enumerate(c_list):
        relay = frac.whole_plan(n_frac, sf, a, b, 30, C=c, max_dose=100)
        relay2 = np.array(relay)[0][:]
        print(relay2)
        #dose_delivery[i] = relay2

if __name__ == '__main__':
    simulation(7)
