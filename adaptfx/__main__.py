import click
import numpy as np
import reinforce.fraction_minimisation as frac
import reinforce.oar_minimisation as oar
import reinforce.tumor_maximisation as tumor
import reinforce.track_tumor_oar as tumor_oar
import common.constants as C
import handler.messages as m
import handler.aft_utils as utils

class RL_object():
    def __init__(self, instruction_filename):
        try: # check if file can be opened
            with open(instruction_filename, 'r') as f:
                read_in = f.read()
            input_dict= eval(read_in)
        except OSError:
            m.aft_error(f'could not open file: "{instruction_filename}"')
        except:
            m.aft_error(f'unknown error while reading file: "{instruction_filename}"')

        try: # check if log flag is existent and boolean
            log_bool = input_dict['log']
        except KeyError:
            m.aft_message('no "log" flag was given, set to "False"')
            log_bool = False
        else:
            if not isinstance(log_bool, bool):
                m.aft_error('"log" flag was not set to boolean')
            else:
                m.logging_init(instruction_filename, log_bool)

        try: # check if algorithm key matches known types
            algorithm = input_dict['algorithm']
        except KeyError:
            m.aft_error(f'"algorithm" key missing in: "{instruction_filename}"')
        else:
            if algorithm not in C.KEY_DICT:
                m.aft_error(f'unknown "algorithm" type: "{algorithm}"')

        try: # check if parameter key exists and is a dictionnary
            parameters = input_dict['parameters']
        except KeyError:
            m.aft_error(f'"parameter" key missing in : "{instruction_filename}"')
        else:
            if not isinstance(parameters, dict):
                m.aft_message_error('"parameters" was not a dictionary')

        m.aft_message('loading keys...', 0)
        whole_dict = utils.key_reader(C.KEY_DICT, C.FULL_DICT, parameters, algorithm)

        self.parameters = whole_dict
        self.algorithm = algorithm
        self.log_bool = log_bool

    def optimise(self):
        params = self.parameters
        if self.algorithm == 'oar':
            relay = oar.whole_plan(
                number_of_fractions=params['number_of_fractions'],
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
            relay = tumor.whole_plan(
                number_of_fractions=params['number_of_fractions'],
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
            relay = frac.whole_plan(
                number_of_fractions=params['number_of_fractions'],
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
            relay = tumor_oar.whole_plan(
                number_of_fractions=params['number_of_fractions'],
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
    m.aft_message_info('log to file:', rl_test.log_bool, 0)
    m.aft_message_info('type of algorithm:', rl_test.algorithm, 0)
    m.aft_message_dict('parameters:', rl_test.parameters, 1)
    m.aft_message('start session...', 1)
    m.aft_message_list('fractionation plan:', rl_test.optimise(), 1)
    m.aft_message('close session...', 1)
    

if __name__ == '__main__':
    main()
