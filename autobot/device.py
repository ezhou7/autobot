import platform

from autobot.utils.fs import read_file


DEVICE_COMPATIBLE_NODE = "/proc/device-tree/compatible"
LINUX_AARCH64 = "Linux-aarch64"


class AutoBotDevice:
    def __init__(self):
        self.os_machine = platform.system() + "-" + platform.machine()

    def get_host(self):
        if self.os_machine == LINUX_AARCH64:
            device_compatible_str = read_file(DEVICE_COMPATIBLE_NODE)
            host = "RK3588" if "rk3588" in device_compatible_str else "RK356x"
        else:
            host = self.os_machine

        return host
