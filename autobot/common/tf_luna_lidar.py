import time
from serial import Serial


class TFLunaLidar:
    """
    Inspiration for this code: https://github.com/makerportal/tfluna-python/blob/main/tfluna_realtime.py
    """
    def __init__(self, baudrate = 57600):
        self.baudrate = baudrate
        self.ser = Serial("/dev/serial0", baudrate, timeout=0)
        self.__buf_size = 9

    def set_sample_rate(self, rate):
        sample_rate_packet = bytearray([0x5a, 0x06, 0x03, rate, 0x00, 0x00])
        self.ser.write(sample_rate_packet)

    def read(self, buf_size):
        correct = True
        counter = self.ser.in_waiting

        if counter < buf_size:
            print("Number of bytes to be read does not match buffer size. Please double check.")
            correct = False

        buf = self.ser.read(buf_size if correct else counter)
        self.ser.reset_input_buffer()

        return buf, correct

    def get_version(self):
        # get version info instruction packet
        cmd = bytearray([0x5a, 0x04, 0x14, 0x00])
        self.ser.write(cmd)
        time.sleep(0.1)

        buf_size = 30
        t0 = time.time()

        while time.time() - t0 < 5:
            buf, correct = self.read(buf_size)
            if not correct:
                return "-1"

            if buf[0] == 0x5a:
                return buf[3:-1].decode("utf-8")
            else:
                self.ser.write(cmd)
                time.sleep(0.1)

    def set_baudrate(self, rate):
        rate2hex = {
            9600: [0x80, 0x25, 0x00],
            19200: [0x00, 0x4b, 0x00],
            38400: [0x00, 0x96, 0x00],
            57600: [0x00, 0xe1, 0x00],
            115200: [0x00, 0xc2, 0x00],
            230400: [0x00, 0x84, 0x00],
            460800: [0x00, 0x08, 0x00],
            921600: [0x00, 0x10, 0x00]
        }
        cmd = bytearray([0x5a, 0x08, 0x06] + rate2hex[rate] + [0x00, 0x00])
        self.ser.write(cmd)
        time.sleep(0.1)
        self.ser.close()
        time.sleep(0.1)
        ser = Serial("/dev/serial0", rate, timeout=0)
        if not ser.is_open:
            ser.open()

        buf_size = 8
        t0 = time.time()

        while time.time() - t0 < 5:
            buf, correct = self.read(buf_size)
            if buf[0] == 0x5a:
                self.baudrate = [r for r, b in rate2hex.items() if b == buf[3:6]][0]
                self.ser = ser
                break

            ser.write(cmd)
            time.sleep(0.1)

        time.sleep(0.1)

    def read_data(self):
        buf_size = 9
        buf, correct = self.read(buf_size)

        if not correct:
            return -1, -1, -1

        # first two bytes are headers
        # second two bytes are distance
        # third two bytes are signal strength
        # fourth two bytes are temperature
        # one more byte is padding (?) not sure here
        if not buf or not len(buf) < self.__buf_size or not buf[0] == 0x59 or not buf[1] == 0x59:
            return -1, -1, -1

        dist = buf[2] + buf[3] * 256
        strength = buf[4] + buf[5] * 256
        temp = buf[6] + buf[7] * 256
        temp = temp / 8 - 256

        return dist, strength, temp
