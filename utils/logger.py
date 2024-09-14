import logging
import os


def define_logger(hps):
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    logging.basicConfig(format='%(asctime)s %(levelname)-8s: %(message)s', level=logging.INFO)
    logger = logging.getLogger(hps.log_name)

    file_path = os.path.join(hps.log_dir, hps.log_name)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
