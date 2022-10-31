# -*- coding: utf-8 -*-
import argparse
import adaptfx.constants as C
import adaptfx.planning as planning
import adaptfx.aft_messages as m
import adaptfx.aft_utils as aft_utils
nme = __name__

class RL_object():
    """
    Reinforcement Learning class to check instructions
    of calculation, invoke keys and define
    calculation settings from file
    """
    def __init__(self, instruction_filename):
        try: # check if file can be opened
            m.aft_message('', nme, 1)
            with open(instruction_filename, 'r') as f:
                read_in = f.read()
            input_dict= eval(read_in)
        except SyntaxError:
            m.aft_error(f'Dictionary Syntax Error in: "{instruction_filename}"', nme)
        except:
            m.aft_error(f'No such file: "{instruction_filename}"', nme)

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
                m.aft_error('invalid "debug" flag was set', nme)

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

        try: # check if keys exists and is a dictionnary
            raw_keys = input_dict['keys']
        except KeyError:
            m.aft_error(f'"keys" is missing in : "{instruction_filename}"', nme)
        else:
            if not isinstance(raw_keys, dict):
                m.aft_error('"keys" is not a dictionary', nme)
            # load and check keys
            m.aft_message('loading keys...', nme, 1)
            keys = aft_utils.key_reader(C.KEY_DICT, C.FULL_DICT, raw_keys, algorithm)
            m.aft_message_dict('keys', keys, nme, 1)

        try: # check if settings exist and is a dictionnary
            user_settings = input_dict['settings']
        except KeyError:
            m.aft_message('no "settings" were given, set to default', nme)
            settings = C.SETTING_DICT
            m.aft_message_dict('settings', settings, nme, 1)
        else:
            if not isinstance(user_settings, dict):
                m.aft_error('"settings" is not a dictionary', nme)
            # load and check settings
            m.aft_message('loading settings...', nme, 1)
            settings = aft_utils.setting_reader(C.SETTING_DICT, user_settings)
            m.aft_message_dict('settings', settings, nme, 1)

        self.algorithm = algorithm
        self.keys = aft_utils.DotDict(keys)
        self.settings = aft_utils.DotDict(settings)

    def optimise(self):
        doses = planning.multiple(self.algorithm, self.keys, self.settings)

        return doses

def main():
    """
    CLI interface to invoke the RL class
    """
    start = aft_utils.timing()
    parser = argparse.ArgumentParser(
        description='Calculate optimal dose per fraction dependent on algorithm type'
    )
    parser.add_argument(
        '-f',
        '--filename',
        metavar='',
        help='input instruction filename of dictionary',
        default=None,
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
    args = parser.parse_args()
    rl_test = RL_object(args.filename)
    m.aft_message('start session...', nme, 1)
    m.aft_message_list('fractionation plan:', rl_test.optimise(), nme, 1)
    aft_utils.timing(start)
    m.aft_message('close session...', nme, 1)


if __name__ == '__main__':
    main()
