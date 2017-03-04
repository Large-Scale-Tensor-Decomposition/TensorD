# Created by ay27 at 17/3/4
import logging
from logging.config import fileConfig

DEFAULT_TYPE = 'DEBUG'
config_file = '../conf/logging_config.ini'
fileConfig(config_file)


def create_logger(level=DEFAULT_TYPE):
    """

    Parameters
    ----------
    level: str
        DEBUG or RELEASE or None

    Returns
    -------
    logger

    """
    return logging.getLogger(level)
