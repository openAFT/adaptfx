import numpy as np
from reinforce import fraction_minimisation as frac
from reinforce import oar_minimisation as oar
from reinforce import tumor_maximisation as tumor

class RL_object():
    def __init__(self, algorithm, **params):
        self.algorithm = algorithm
        parameters = {}
        list = ['number_of_fractions',
                'sparing_factors',
                'alpha',
                'beta',
                'goal',
                'C',
                'bound_OAR',
                'bound_tumor',
                'BED_OAR',
                'BED_tumor',
                'OAR_limit',
                'min_dose',
                'max_dose',
                'fixed_prob',
                'fixed_mean',
                'fixed_std',

        ]

        # for key in list:
        #     if key in params:
        #         parameters[key] = params['alpha']
        #     else:
        #         if self.algorithm=='frac' and  key in list_frac:
        #             parameters[key] = params['alpha']
        #         elif self.algorithm=='track' and key in list_track:
        #             parameters[key] = params['alpha']
        #         else:
        #             print("key", key, "is missing in input")

        for key, value in params.items():
            if key in list:
                parameters[key] = value
            else:
                print("key", key, "is not allowed")
        self.parameters = parameters

    def optimise(self):
        relay = frac.whole_plan(self.parameters)
        print(relay)

if __name__ == '__main__':
    rl_test = RL_object(algorithm='frac',
                        number_of_fractions=8,
                        sparing_factors=np.linspace(1,1.1,9),
                        alpha=2.7,
                        beta=0.014,
                        goal=30,
                        C=3
    )
    # rl_test.optimise()
