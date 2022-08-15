import click
import numpy as np
from reinforce import fraction_minimisation as frac
from reinforce import oar_minimisation as oar
from reinforce import tumor_maximisation as tumor
from reinforce import track_tumor_oar as tumor_oar
from common import constants as C

class RL_object():
    def __init__(self, input_dict):
        algorithm = input_dict['algorithm']
        parameters = input_dict['parameters']
        full_dict = C.FULL_DICT
        key_dict = C.KEY_DICT
        whole_dict = {}

        for key in key_dict[algorithm]:
            if key in parameters:
                whole_dict[key] = parameters[key]
            elif key not in parameters and key in full_dict:
                if full_dict[key] == None:
                    print(f'mandatory key "{key}" is missing')
                else:
                    whole_dict[key] = full_dict[key]

        for key in parameters:
            if key not in key_dict[algorithm] and key in full_dict:
                print(f'key "{key}" is not allowed for "{algorithm}", is not passed')
            elif key not in full_dict:
                print(f'key "{key}" is invalid, is not passed')

        print(whole_dict)

        print("Successfully loaded keys")

        self.parameters = whole_dict
        self.algorithm = algorithm

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

        elif self.algorithm == 'tumor_oar':
            relay = tumor_oar.whole_plan(number_of_fractions=params['number_of_fractions'],
                                        sparing_factors=params['sparing_factors'],
                                        alpha=params['alpha'],
                                        beta=params['beta'],
                                        bound_OAR=params['bound_OAR'],
                                        bound_tumor=['bound_tumor'],
                                        abt=params['abt'],
                                        abn=params['abn'],
                                        min_dose=params['min_dose'],
                                        max_dose=params['max_dose'],
                                        fixed_prob=params['fixed_prob'],
                                        fixed_mean=params['fixed_mean'],
                                        fixed_std=params['fixed_std'],
            )

        else:
            print(f'algorithm "{self.algorithm}" not known')

        return np.array(relay,dtype=object)[0][:]

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('instruction_filename')
@click.option('--gui', '-g', default=False,
        help='Provide Graphic User Interface for planning')

def main(instruction_filename, gui):
    '''
    \b
    Calculate optimal dose per fraction dependent on algorithm type

    \b
    <instruction_filename>   : input instruction filename
    '''
    with open(instruction_filename, 'r') as f:
        read_in = f.read()
        output_dict= eval(read_in)
    print(output_dict)

    rl_test = RL_object(output_dict).optimise()

    print(rl_test)
    
    # print(rl_test.optimise())

if __name__ == '__main__':
    main()
