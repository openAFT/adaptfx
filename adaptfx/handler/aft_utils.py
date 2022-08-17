import time
import handler.messages as m

def timing_with_time():
    start = time.perf_counter()
    time.sleep(1)
    stop = time.perf_counter()
    return (stop - start)

def key_reader(all_keys, full_dict, parameters, algorithm):
    whole_dict = {}
    key_dict = all_keys[algorithm]

    for key in key_dict:
        if key in parameters:
            whole_dict[key] = parameters[key]
        elif key not in parameters:
            if full_dict[key] == None:
                m.aft_error(f'missing mandatory key: "{key}"')
            else:
                whole_dict[key] = full_dict[key]

    for key in parameters:
        if key not in key_dict and key in full_dict:
            m.aft_warning(
                f'key: "{key}" is not allowed for "{algorithm}"', 0
                )
        elif key not in full_dict:
            m.aft_warning(f'unexpected parameter: "{key}" invalid', 0)

    return whole_dict