import sys
import logging
import tensorflow as tf


def initialize_logging(file_name, override_tf=True):
    log_format = logging.Formatter('%(asctime)s : %(message)s')
    logger = logging.getLogger()
    logger.handlers = []

    if override_tf:
        logger_tf = logging.getLogger('tensorflow')
        logger_tf.setLevel(logging.INFO)
        logger_tf.handlers = []

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


initialize_logging('log', override_tf=False)
tf.logging.info('message1')

initialize_logging('log', override_tf=True)
tf.logging.info('message2')
