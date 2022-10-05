import sys
import logging
import os

def logging_init(filename, log, debug):
    """
    log initialisation to write to filename

    Parameters
    ----------
    filename : string
        filename of log file
    log : bool
        if true store log to filename
    debug: bool
        if true log extensive message

    Returns
    -------
    None
        
    """
    if debug:
        format_file = '%(asctime)s [%(levelname)s] [%(name)s]: %(message)s'
        format_out = 'AFT> [%(levelname)s] [%(name)s]: %(message)s'
        log_level = logging.DEBUG
    else:
        format_file = '%(asctime)s [%(levelname)s]: %(message)s'
        format_out = 'AFT> [%(levelname)s]: %(message)s'
        log_level = logging.INFO
    
    if log:
        # create logfile name
        basename = "{0}".format(*filename.rsplit('.', 1))
        # search for existing filename ...
        i = 1
        while os.path.exists("{0}_{1}.{2}".format(basename, i, 'log')):
            # exponential search if many files exist
            i *= 2
        a, b = (i // 2, i)
        while a+1 < b:
            c = a + b // 2
            if os.path.exists("{0}_{1}.{2}".format(basename, c, 'log')):
                a, b = (c, b)
            else:
                a, b = (a,c)
        # ... end of search
        log_filename = "{0}_{1}.{2}".format(basename, b, 'log')
        logging.basicConfig(
            format=format_file,
            level=log_level, 
            datefmt='%d-%m-%Y %H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
                ]
        )
    else:
        logging.basicConfig(
            format=format_out,
            level=log_level, 
            handlers=[
                logging.StreamHandler(sys.stdout)
                ]
        )

def aft_error(error, name):
    """
    error message that exits process

    Parameters
    ----------
    error : string
        message that is logged on logging.ERROR level
    name : string
        logger name

    Returns
    -------
    None
        
    """
    log_name = logging.getLogger(name)
    print('AFT>')
    log_name.error(error)
    log_name.info('exiting Session...')
    sys.exit()

def aft_warning(warning, name, mode=0):
    """
    warning message

    Parameters
    ----------
    warning : string
        message that is logged in logging.WARNING level
    name : string
        logger name

    Returns
    -------
    None
        
    """
    log_name = logging.getLogger(name)
    if mode == 1:
        print('AFT>')
    log_name.warning(warning)

def aft_message(message, name, mode=0):
    """
    information message

    Parameters
    ----------
    message : string
        message that is logged on logging.INFO level
    name : string
        logger name

    Returns
    -------
    None
        
    """
    log_name = logging.getLogger(name)
    if mode == 1:
        print('AFT>')
    log_name.info(message)

def aft_message_info(message, info, name, mode=0):
    """
    information message belonging to certain string

    Parameters
    ----------
    message : string
        message that is logged on logging.INFO level
    info : string
        variable information
    name : string
        logger name

    Returns
    -------
    None
        
    """
    log_name = logging.getLogger(name)
    if mode == 1:
        print('AFT>')
    log_name.info(f'{message} {info}')

def aft_message_dict(message, dict, name, mode=0):
    """
    print statement for dictionary

    Parameters
    ----------
    message : string
        message that is logged on logging.INFO level
    dict : dict
        dictionary of parameters
    name : string
        logger name

    Returns
    -------
    None
        
    """
    log_name = logging.getLogger(name)
    if mode == 1:
        print('AFT>')
    log_name.info(message)
    for key, value in dict.items():
        log_name.info(f'|{key: <19}| {value}')
        
def aft_message_list(message, struct, name, mode=0):
    """
    print statement for general structures (e.g. list)

    Parameters
    ----------
    message : string
        message that is logged on logging.INFO level
    struct : list
        list of fractionation plans
    name : string
        logger name

    Returns
    -------
    None
        
    """
    log_name = logging.getLogger(name)
    if mode == 1:
        print('AFT>')
    log_name.info(message)
    log_name.info('%s', struct)
