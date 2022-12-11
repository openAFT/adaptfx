# -*- coding: utf-8 -*-
import time
from adaptfx import aft_message, aft_error, aft_warning, aft_message
nme = __name__

def stat_rounding(number, decimal):
    magnitude = 10 ** decimal
    return round(number * magnitude) / magnitude

def timing(start=None):
    """
    measure time for general process:
    for start: var_name = timing()
    for strop: timing(var_name)

    Parameters
    ----------
    start : time of starting function
        the default is None.

    Returns
    -------
    start : float
        starting time
        
    """
    if start == None:
        start_time = time.perf_counter()
        return start_time
    else:
        stop = time.perf_counter()
        time_elapsed = stat_rounding((stop - start), 4)
        aft_message(f'process duration: {time_elapsed} s:', nme, 1)

def key_reader(all_keys, full_dict, user_keys, algorithm):
    """
    read and check all keys from a instruction
    file by cycling through all_keys

    Parameters
    ----------
    all_keys : dict
        all keys necessary for some algorithm type.
    full_dict : dict
        all possible keys.
    user_keys : dict
        read in keys from an instruction file.
    algorithm : string
        type of algorithm.

    Returns
    -------
    whole dict : dict
        all keys copied from parameters
        
    """
    aft_message('loading keys...', nme, 1)
    whole_dict = full_dict.copy()
    key_dict = all_keys[algorithm]

    for key in key_dict:
        if key in user_keys:
            whole_dict[key] = user_keys[key]
        elif key not in user_keys:
            if full_dict[key] == None:
                aft_error(f'missing mandatory key: "{key}"', nme)
            else:
                whole_dict[key] = full_dict[key]

    for key in user_keys:
        if key not in key_dict and key in full_dict:
            aft_warning(
                f'key: "{key}" is not allowed for "{algorithm}"', nme, 0
                )
        elif key not in full_dict:
            aft_warning(f'unexpected key: "{key}"', nme, 0)

    return dict((k, whole_dict[k]) for k in key_dict)

def setting_reader(all_settings, user_settings):
    """
    read and check all keys from a settings
    by cycling through all_settings

    Parameters
    ----------
    all_settings : dict
        all settings necessary for calculation
    settings : dict
        all settings for calculation

    Returns
    -------
    whole_settings : dict
        all settings
        
    """
    aft_message('loading settings...', nme, 1)
    whole_settings = all_settings.copy()

    for skey in all_settings:
        if skey in user_settings:
            whole_settings[skey] = user_settings[skey]
    
    for skey in user_settings:
        if skey not in all_settings:
            aft_warning(f'unexpected setting: "{skey}"', nme, 0)

    return whole_settings

class DotDict(dict):
    """
    object with nested dot.notation to
    access dictionary attributes
    """
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super(DotDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        return self.get(key)
    
    def __delitem__(self, key):
        # found = self.get(key, DotDict.MARKER)
        # if found is DotDict.MARKER:
        super(DotDict, self).__delitem__(key)

    __setattr__, __getattr__, __delattr__ = __setitem__, __getitem__, __delitem__