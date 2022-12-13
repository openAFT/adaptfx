# -*- coding: utf-8 -*-
import argparse
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
            input_dict= eval(read_in)
        except TypeError:
            if isinstance(instruction_filename, dict):
                input_dict = instruction_filename
            else:
                afx.aft_error(f'"{instruction_filename}" not a filename or dict', nme)
        except SyntaxError:
            afx.aft_error(f'Dictionary Syntax Error in: "{instruction_filename}"', nme)
        except OSError:
            afx.aft_error(f'No such file: "{instruction_filename}"', nme)

        try: # check if log flag is existent and boolean
            log_bool = input_dict['log']
        except KeyError:
            afx.aft_warning('no "log" flag was given, set to 0', nme)
            log_bool = 0
        else:
            if not log_bool in [0,1]:
                afx.aft_error('invalid "log" flag was set', nme)

        try: # check if log flag is existent and boolean
            log_level = input_dict['level']
        except KeyError:
            afx.aft_warning('no "level" mode was given, set to 1', nme)
            log_level = 1
        else:
            if not log_level in [0,1,2]:
                afx.aft_error('invalid "debug" flag was set', nme)

        afx.logging_init(instruction_filename, log_bool, log_level)
        afx.aft_message_info('log level:', log_level, nme)
        afx.aft_message_info('log to file:', log_bool, nme)

        try: # check if algorithm key matches known types
            algorithm = input_dict['algorithm']
        except KeyError:
            afx.aft_error(f'"algorithm" key missing in: "{instruction_filename}"', nme)
        else:
            if algorithm not in afx.KEY_DICT:
                afx.aft_error(f'unknown "algorithm" type: "{algorithm}"', nme)
            else:
                afx.aft_message_info('algorithm:', algorithm, nme)

        try: # check if keys exists and is a dictionnary
            raw_keys = input_dict['keys']
        except KeyError:
            afx.aft_error(f'"keys" is missing in : "{instruction_filename}"', nme)
        else:
            if not isinstance(raw_keys, dict):
                afx.aft_error('"keys" is not a dictionary', nme)
            # load and check keys
            keys = afx.key_reader(afx.KEY_DICT, afx.FULL_DICT, raw_keys, algorithm)
            afx.aft_message_dict('keys', keys, nme, 1)

        try: # check if settings exist and is a dictionnary
            user_settings = input_dict['settings']
        except KeyError:
            afx.aft_warning('no "settings" were given, set to default', nme, 1)
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
    
    def plot(self):
        sf = self.output.sf
        states = self.output.states
        if self.settings.plot_policy:
            afx.plot_val(sf, states, self.output.policy, self.output.policy_list)
        if self.settings.plot_values:
            afx.plot_val(sf, states, self.output.value, self.output.values_list)
        if self.settings.plot_remains:
            afx.plot_val(sf, states, self.output.remains, self.output.remains_list)

        if (self.settings.plot_policy + self.settings.plot_values + self.settings.plot_remains):
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
    afx.aft_message_list('physical dose:', plan.output.physical_doses, nme, 1)
    afx.aft_message_list('tumor dose:', plan.output.tumor_doses, nme)
    afx.aft_message_list('oar dose:', plan.output.oar_doses, nme)

    # show accumulated dose
    afx.aft_message_info('accumulated oar dose:', plan.output.oar_sum, nme, 1)
    afx.aft_message_info('accumulated tumor dose:', plan.output.tumor_sum, nme)
    plan.plot()
    afx.aft_message('close session...', nme, 1)


if __name__ == '__main__':
    main()
