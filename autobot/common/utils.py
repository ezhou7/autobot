import logging
import os
import time

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


def get_wsl_host_ip():
    """
    In a Linux system, returns the IP of the host Windows 
    computer on the internal virtual network.
    """
    ip = ""
    if not os.path.exists("/etc/resolv.conf"):
        return ip

    with open("/etc/resolv.conf") as file:
        for line in file:
            if "nameserver" in line:
                ip = line.split()[-1]
                break
    return ip


async def measure(func, time_list: list, is_async: bool, *args):
    """Execute a function and measure the time that it takes to run.
    
    Adds the time to the end of the list of values provided in time_list."""
    start_time = time.perf_counter()

    if is_async:
        result = await func(*args)
    else:
        result = func(*args)
    end_time = time.perf_counter()
    time_list.append(end_time - start_time)

    return result
