import interpol2D_OARminfrac as intmin

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma, truncnorm

[a, b] = intmin.data_fit(np.array([[0.99, 0.95, 0.98, 1.02], [0.95, 0.9, 0.8, 0.9]]))
relay = intmin.whole_plan(4, [0.99, 0.95, 0.98, 0.96, 1.02], a, b, 50)
print(relay)