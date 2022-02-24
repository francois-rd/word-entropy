import logging


def warn_not_empty(kwargs):
    """
    Logs a warning if the given kwargs is not empty.
    """
    if kwargs:
        logging.warning("Unexpected kwargs: {}".format(list(kwargs.keys())))
