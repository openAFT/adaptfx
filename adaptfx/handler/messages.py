import sys
import logging

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
    print('AFT>')
    logging.error(error)
    logging.info('exiting Session...')
    sys.exit()

def aft_warning(warning, mode=0):
    if mode == 0:
        logging.warning(warning)
    if mode == 1:
        print('AFT>')
        logging.warning(warning)

def aft_message(message, mode=0):
    if mode == 0:
        logging.info(message)
    elif mode == 1:
        print('AFT>')
        logging.info(message)

def aft_message_info(message, info, mode=0):
    if mode == 0:
        logging.info(f'{message} {info}')
    elif mode == 1:
        print('AFT>')
        logging.info(f'{message} {info}')

def aft_message_dict(message, dict, mode=0):
    if mode == 0:
        logging.info(message)
        for key, value in dict:
            logging.info('{%(key)s: %(value)s}' %
                {'key': key, "value": value})
    if mode == 1:
        print('AFT>')
        logging.info(message)
        for key, value in dict.items():
            logging.info('{%(key)s: %(value)s}' %
                {'key': key, "value": value})
        
def aft_message_list(message, struct, mode=0):
    if mode == 0:
        logging.info(message)
        logging.info('%s', struct)
    elif mode == 1:
        print('AFT>')
        logging.info(message)
        logging.info('%s', struct)
