import os
import time


from autobot.utils.fs import read_file


class Properties:
    def __init__(self, props: dict):
        self.__dict__.update(props)


def centroid(x1: int, y1: int, x2: int, y2: int):
    xc = x1 + ((x2 - x1) >> 1)
    yc = y1 + ((y2 - y1) >> 1)

    return xc, yc


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

