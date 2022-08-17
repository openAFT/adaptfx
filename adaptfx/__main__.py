import click
import numpy as np
from reinforce import fraction_minimisation as frac
from reinforce import oar_minimisation as oar
from reinforce import tumor_maximisation as tumor
from reinforce import track_tumor_oar as tumor_oar
from common import constants as C
from handler import messages as m

class RL_object():
    def __init__(self, instruction_filename):
        try:
            with open(instruction_filename, 'r') as f:
                read_in = f.read()
            input_dict= eval(read_in)
        except:
            m.logging_init('__fail__.log', False)
            m.aft_error(f'Cannot read file: "{instruction_filename}"')

        algorithm = input_dict['algorithm']
        parameters = input_dict['parameters']
        logging = input_dict['logging']
        full_dict = C.FULL_DICT
        key_dict = C.KEY_DICT
        whole_dict = {}

        m.logging_init(instruction_filename, logging)

        m.aft_message('read Instructions...', 0)

        for key in key_dict[algorithm]:
            if key in parameters:
                whole_dict[key] = parameters[key]
            elif key not in parameters:
                if full_dict[key] == None:
                    m.aft_error(f'missing mandatory key: "{key}"')
                else:
                    whole_dict[key] = full_dict[key]

        for key in parameters:
            if key not in key_dict[algorithm] and key in full_dict:
                m.aft_warning(f'key: "{key}" is not allowed for "{algorithm}", is not passed', 0)
            elif key not in full_dict:
                m.aft_warning(f'key: "{key}" is invalid, is not passed', 0)

        m.aft_message('success, loading keys', 1)

        self.parameters = whole_dict
        self.algorithm = algorithm
        self.logging = logging

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
    rl_test = RL_object(instruction_filename)
    m.aft_message_info('log to file:', rl_test.logging, 0)
    m.aft_message_info('type of algorithm:', rl_test.algorithm, 0)
    m.aft_message_dict('keys:', rl_test.parameters, 1)
    m.aft_message('start session...', 1)
    m.aft_message_list('fractionation plan:', rl_test.optimise(), 1)
    m.aft_message('close session...', 1)
    

if __name__ == '__main__':
    main()
