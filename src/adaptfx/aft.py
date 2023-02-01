# -*- coding: utf-8 -*-
import argparse
import json
import adaptfx as afx
import sys
nme = __name__

class RL_object():
    """
    Reinforcement Learning class to check instructions
    of calculation, invoke keys and define
    calculation settings from file
    """
    def __init__(self, instruction_filename):
        try: # check if file can be opened
            with open(instruction_filename, 'r') as f:
                read_in = f.read()
            input_dict= json.loads(read_in)
        except TypeError:
            if isinstance(instruction_filename, dict):
                input_dict = instruction_filename
            else:
                afx.aft_error(f'"{instruction_filename}" not a filename or dict', nme)
        except SyntaxError as syntax_err:
            afx.aft_error(f'error in "{instruction_filename}", {syntax_err}', nme)
        except OSError:
            afx.aft_error(f'No such file: "{instruction_filename}"', nme)
        except ValueError as decode_err:
            afx.aft_error(f'decode error in "{instruction_filename}", {decode_err}', nme)
        except:
            afx.aft_error(f'error in "{instruction_filename}", {sys.exc_info()}', nme)

        try: # check if log flag is existent and boolean
            log_bool = input_dict['log']
        except KeyError as log_err:
            log_bool = afx.LOG_BOOL
            afx.aft_warning(f'no {log_err} flag was given, set to {log_bool}', nme)
        else:
            if not log_bool in afx.LOG_BOOL_LIST:
                afx.aft_error('invalid "log" flag was set', nme)

        try: # check if log flag is existent and boolean
            log_level = input_dict['level']
        except KeyError as level_err:
            log_level = afx.LOG_LEVEL
            afx.aft_warning(f'no {level_err} mode was given, set to {log_level}', nme)
        else:
            if not log_level in afx.LOG_LEVEL_LIST:
                afx.aft_error('invalid "debug" flag was set', nme)

        afx.logging_init(instruction_filename, log_bool, log_level)
        afx.aft_message_info('log level:', log_level, nme)
        afx.aft_message_info('log to file:', log_bool, nme)

        try: # check if algorithm key matches known types
            algorithm = input_dict['algorithm']
        except KeyError as algo_err:
            afx.aft_error(f'{algo_err} key missing in: "{instruction_filename}"', nme)
        else:
            if algorithm not in afx.KEY_DICT:
                afx.aft_error(f'unknown "algorithm" type: "{algorithm}"', nme)
            else:
                afx.aft_message_info('algorithm:', algorithm, nme)

        try: # check if keys exists and is a dictionnary
            raw_keys = input_dict['keys']
        except KeyError as key_err:
            afx.aft_error(f'{key_err} is missing in : "{instruction_filename}"', nme)
        else:
            if not isinstance(raw_keys, dict):
                afx.aft_error('"keys" is not a dictionary', nme)
            # load and check keys
            keys = afx.key_reader(afx.KEY_DICT, afx.FULL_DICT, raw_keys, algorithm)
            afx.aft_message_dict('keys', keys, nme, 1)

        try: # check if settings exist and is a dictionnary
            user_settings = input_dict['settings']
        except KeyError as sets_err:
            afx.aft_warning(f'no {sets_err} were given, set to default', nme, 1)
            settings = afx.SETTING_DICT
            afx.aft_message_dict('settings', settings, nme, 1)
        else:
            if not isinstance(user_settings, dict):
                afx.aft_error('"settings" is not a dictionary', nme)
            # load and check settings
            settings = afx.setting_reader(afx.SETTING_DICT, user_settings)
            afx.aft_message_dict('settings', settings, nme, 1)

        self.algorithm = algorithm
        self.log = log_bool
        self.log_level = log_level
        self.keys = afx.DotDict(keys)
        self.settings = afx.DotDict(settings)

    def optimise(self):
        self.output = afx.multiple(self.algorithm, self.keys, self.settings)

    def fraction_counter(self):
        if self.algorithm == 'frac' and self.keys.fraction == 0:
            # in the case for whole treatment calculations
            n_frac = self.keys.number_of_fractions
            afx.aft_message(f'fractions used: ({self.output.fractions_used}/{n_frac})', nme)
            
    def plot(self):
        out = self.output
        sets = self.settings
        if self.settings.plot_policy:
            afx.plot_val(out.policy.sf, out.policy.states, out.policy.val, out.policy.fractions)
        if self.settings.plot_values:
            afx.plot_val(out.value.sf, out.value.states, out.value.val, out.value.fractions)
        if self.settings.plot_remains:
            afx.plot_val(out.remains.sf, out.remains.states, out.remains.val, out.remains.fractions)

        if sets.plot_policy or sets.plot_values or sets.plot_remains:
            afx.show_plot()
        else:
            afx.aft_message('nothing to plot', nme, 1)

def main():
    """
    CLI interface to invoke the RL class
    """
    start = afx.timing()
    parser = argparse.ArgumentParser(
        description='Calculate optimal dose per fraction dependent on algorithm type'
    )
    parser.add_argument(
        '-f',
        '--filename',
        metavar='',
        help='input instruction filename of dictionary',
        type=str
    )
    parser.add_argument(
        '-g',
        '--gui',
        metavar='',
        help='Provide Graphic User Interface for planning - NOT YET AVAILABLE',
        default=False,
        type=bool
    )
    # In case there is no input show help
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    plan = RL_object(args.filename)

    afx.aft_message('start session...', nme, 1)
    plan.optimise()
    afx.timing(start)

    # show retrospective dose prescribtion
    afx.aft_message_list('physical tumor dose:', plan.output.physical_doses, nme, 1)
    afx.aft_message_list('tumor dose:', plan.output.tumor_doses, nme)
    afx.aft_message_list('oar dose:', plan.output.oar_doses, nme)

    # show accumulated dose
    afx.aft_message_info('accumulated oar dose:', plan.output.oar_sum, nme, 1)
    afx.aft_message_info('accumulated tumor dose:', plan.output.tumor_sum, nme)

    # number of fractions used
    plan.fraction_counter()

    # show plots
    plan.plot()
    afx.aft_message('close session...', nme, 1)


if __name__ == '__main__':
    main()
