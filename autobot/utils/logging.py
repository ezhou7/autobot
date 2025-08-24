import logging


LOGGING_FORMAT = '%(levelname)s:%(name)s: %(message)s'
SYSTEM_INFO_FORMATTER = '%(asctime)s,%(message)s'


def stdout_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a dedicated logger for a module."""
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False

    if len(log.handlers) == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
        log.addHandler(handler)

    return log
