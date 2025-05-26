
import socket
import fcntl
import struct

def get_wlan0_ip(interface):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        iface_bytes = interface.encode("utf-8")[:15]
        ip_addr = fcntl.ioctl(
            sock.fileno(),
            0x8915,
            struct.pack(
                '256s',
                iface_bytes + b'\x00' * (256 - len(iface_bytes))
            )
        )
        ip_addr = ip_addr[20:24]
        return socket.inet_ntoa(ip_addr)
    except OSError as e:
        print(e)
        return None