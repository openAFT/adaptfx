import sys
import logging
import time

def logging_init(filename, switch):
    if switch:
        log_filename = "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + ['log'])
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s]: %(message)s',
            level=logging.INFO, 
            datefmt='%d-%m-%Y %H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
                ]
        )
    else:
        logging.basicConfig(
            format='AFT> [%(levelname)s]: %(message)s',
            level=logging.INFO, 
            handlers=[
                logging.StreamHandler(sys.stdout)
                ]
        )

def aft_error(error):
    print('AFT> ')
    logging.error(f'{error}')
    print('AFT> Exiting Session...')
    print('AFT> ')
    sys.exit()

def aft_warning(warning, mode=0):
    if mode == 0:
        print('AFT>')
        logging.warning(warning)
    if mode == 1:
        print('AFT> ')
        logging.warning(warning)
        print('AFT> ')

def aft_message(message, mode=0):
    if mode == 0:
        logging.info(message)
    elif mode == 1:
        print('AFT> ')
        logging.info(message)
    elif mode ==2:
        print('AFT> ')
        logging.info(message)
        print('AFT> ')

def aft_message_struct(message, struct, mode):
    if mode == 0:
        print(f'AFT> {message} {struct}')
    elif mode == 1:
        print('AFT> ')
        print(f'AFT> {message}')
        print(f'AFT> {struct}')
    elif mode == 2:
        print('AFT> ')
        print(f'AFT> {message}')
        print(f'AFT> {struct}')
        print('AFT> ')

def timing_with_time():
    start = time.perf_counter()
    time.sleep(1)
    stop = time.perf_counter()
    return (stop - start)
