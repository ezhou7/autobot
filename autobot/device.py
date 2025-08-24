import platform

from autobot.utils.fs import read_file
from autobot.common.utils import stdout_logger


DEVICE_COMPATIBLE_NODE = "/proc/device-tree/compatible"
LINUX_AARCH64 = "Linux-aarch64"

logger = stdout_logger(__name__)


def get_host():
    os_machine = platform.system() + "-" + platform.machine()
    host = os_machine

    if os_machine == LINUX_AARCH64:
        try:
            device_compatible_str = read_file(DEVICE_COMPATIBLE_NODE)
            host = "RK3588" if "rk3588" in device_compatible_str else "RK356x"
        except FileNotFoundError:
            logger.info(f"Device={os_machine} likely not RKNN native")
            pass

    return host
