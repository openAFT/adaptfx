import numpy as np
from reinforce import fraction_minimisation as frac
from reinforce import oar_minimisation as oar
from reinforce import tumor_maximisation as tumor

class RL_object():
    def __init__(self, algorithm, **params):
        self.algorithm = algorithm

        dict = {'number_of_fractions':None,
                'sparing_factors':None,
                'alpha':None,
                'beta':None,
                'goal':None,
                'C':None,
                'bound_OAR':None,
                'bound_tumor':None,
                'BED_OAR':None,
                'BED_tumor':None,
                'OAR_limit':None,
                'abt':10,
                'abn':3,
                'min_dose':0,
                'max_dose':22.3,
                'fixed_prob':0,
                'fixed_mean':0,
                'fixed_std':0,
        }

        for key, value in params.items():
            if key in dict:
                dict[key] = value
            else:
                print("key '",key,"' is not allowed")

        self.parameters = dict

    def optimise(self):
        params = self.parameters
        if self.algorithm == 'oar':
            relay = oar.whole_plan(number_of_fractions=params['number_of_fractions'],
                                   sparing_factors=params['sparing_factors'],
                                   alpha=params['alpha'],
                                   beta=params['beta'],
                                   goal=params['goal'],
                                   abt=params['abt'],
                                   abn=params['abn'],
                                   min_dose=params['min_dose'],
                                   max_dose=params['max_dose'],
                                   fixed_prob=params['fixed_prob'],
                                   fixed_mean=params['fixed_mean'],
                                   fixed_std=params['fixed_std'],
            )
        elif self.algorithm == 'tumor':
            relay = tumor.whole_plan(number_of_fractions=params['number_of_fractions'],
                                    sparing_factors=params['sparing_factors'],
                                    alpha=params['alpha'],
                                    beta=params['beta'],
                                    OAR_limit=params['OAR_limit'],
                                    abt=params['abt'],
                                    abn=params['abn'],
                                    min_dose=params['min_dose'],
                                    max_dose=params['max_dose'],
                                    fixed_prob=params['fixed_prob'],
                                    fixed_mean=params['fixed_mean'],
                                    fixed_std=params['fixed_std'],
            )
        elif self.algorithm == 'frac':
            relay = frac.whole_plan(number_of_fractions=params['number_of_fractions'],
                                    sparing_factors=params['sparing_factors'],
                                    alpha=params['alpha'],
                                    beta=params['beta'],
                                    goal=params['goal'],
                                    C=params['C'],
                                    abt=params['abt'],
                                    abn=params['abn'],
                                    min_dose=params['min_dose'],
                                    max_dose=params['max_dose'],
                                    fixed_prob=params['fixed_prob'],
                                    fixed_mean=params['fixed_mean'],
                                    fixed_std=params['fixed_std'],
            )
        else:
            print("algorithm", self.algorithm, "not known")

        return relay

if __name__ == '__main__':
    rl_test = RL_object(algorithm='frac',
                        number_of_fractions=8,
                        sparing_factors=np.linspace(1,1.1,9),
                        alpha=2.7,
                        beta=0.014,
                        goal=30,
                        C=3
    )
    
    rl_test.optimise()
