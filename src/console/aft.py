# -*- coding: utf-8 -*-

import click
import numpy as np
import reinforce.fraction_minimisation as frac
import reinforce.oar_minimisation as oar
import reinforce.tumor_maximisation as tumor
import reinforce.track_tumor_oar as tumor_oar
import common.constants as C
import handler.messages as m
import handler.aft_utils as utils
nme = __name__

class RL_object():
    def __init__(self, instruction_filename):
        try: # check if file can be opened
            m.aft_message('', nme, 1)
            with open(instruction_filename, 'r') as f:
                read_in = f.read()
            input_dict= eval(read_in)
        except:
            m.aft_error(f'could not open file: "{instruction_filename}"', nme)

        try: # check if log flag is existent and boolean
            log_bool = input_dict['log']
        except KeyError:
            m.aft_message('no "log" flag was given, set to 0', nme)
            log_bool = 0
        else:
            if not log_bool in [0,1]:
                m.aft_error('invalid "log" flag was set', nme)

        try: # check if log flag is existent and boolean
            debug_bool = input_dict['debug']
        except KeyError:
            m.aft_message('no "debug" flag was given, set to 0', nme)
            debug_bool = 0
        else:
            if not debug_bool in [0,1]:
                m.aft_error('invalid "debug" flag was set')

        m.logging_init(instruction_filename, log_bool, debug_bool)
        m.aft_message_info('debug mode:', debug_bool, nme, 0)
        m.aft_message_info('log to file:', log_bool, nme, 0)

        try: # check if algorithm key matches known types
            algorithm = input_dict['algorithm']
        except KeyError:
            m.aft_error(f'"algorithm" key missing in: "{instruction_filename}"', nme)
        else:
            if algorithm not in C.KEY_DICT:
                m.aft_error(f'unknown "algorithm" type: "{algorithm}"', nme)
            else:
                m.aft_message_info('algorithm:', algorithm, nme, 0)

        try: # check if parameter key exists and is a dictionnary
            parameters = input_dict['parameters']
        except KeyError:
            m.aft_error(f'"parameter" key missing in : "{instruction_filename}"', nme)
        else:
            if not isinstance(parameters, dict):
                m.aft_message_error('"parameters" was not a dictionary', nme)

        m.aft_message('loading keys...', nme, 1)
        whole_dict = utils.key_reader(C.KEY_DICT, C.FULL_DICT, parameters, algorithm)
        m.aft_message_dict('parameters:', whole_dict, nme, 1)

        self.parameters = whole_dict
        self.algorithm = algorithm

    def optimise(self):
        params = self.parameters
        if self.algorithm == 'oar':
            relay = oar.whole_plan(
                number_of_fractions=params['number_of_fractions'],
                sparing_factors=params['sparing_factors'],
                alpha=params['alpha'],
                beta=params['beta'],
                goal=params['tumor_goal'],
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
                goal=params['tumor_goal'],
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
                bound_OAR=params['OAR_limit'],
                bound_tumor=params['tumor_goal'],
                abt=params['abt'],
                abn=params['abn'],
                min_dose=params['min_dose'],
                max_dose=params['max_dose'],
                fixed_prob=params['fixed_prob'],
                fixed_mean=params['fixed_mean'],
                std_fixed=params['fixed_std'],
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
    start = utils.timing()
    rl_test = RL_object(instruction_filename)
    m.aft_message('start session...', nme, 1)
    m.aft_message_list('fractionation plan:', rl_test.optimise(), nme, 1)
    utils.timing(start)
    m.aft_message('close session...', nme, 1)

    

if __name__ == '__main__':
    main()
