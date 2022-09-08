import time
import handler.messages as m
nme = __name__

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
        time_elapsed = (stop - start)
        m.aft_message(f'process duration: {time_elapsed} s:', nme, 1)

def key_reader(all_keys, full_dict, parameters, algorithm):
    """
    read and check all keys from a parameters
    file by cycling through all_keys

    Parameters
    ----------
    all_keys : dict
        all keys necessary for some algorithm type.
    full_dict : dict
        all possible keys.
    parameters : dict
        read in parameters from an instruction file.
    algorithm : string
        type of algorithm.

    Returns
    -------
    whole dict : dict
        all keys copied from parameters
        
    """
    whole_dict = full_dict.copy()
    key_dict = all_keys[algorithm]

    for key in key_dict:
        if key in parameters:
            whole_dict[key] = parameters[key]
        elif key not in parameters:
            if full_dict[key] == None:
                m.aft_error(f'missing mandatory key: "{key}"', nme)
            else:
                whole_dict[key] = full_dict[key]

    for key in parameters:
        if key not in key_dict and key in full_dict:
            m.aft_warning(
                f'key: "{key}" is not allowed for "{algorithm}"', nme, 0
                )
        elif key not in full_dict:
            m.aft_warning(f'unexpected parameter: "{key}" invalid', nme, 0)

    return whole_dict