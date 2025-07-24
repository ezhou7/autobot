# SPDX-FileCopyrightText: 2018 ladyada for Adafruit Industries
# SPDX-FileCopyrightText: 2022 johnrbnsn
#
# SPDX-License-Identifier: MIT

"""
`lidarlite`
====================================================

A Python library for Garmin LIDAR Lite sensors over I2C, converted for use on Orange Pi with standard Python.

This version uses smbus2 for I2C communication instead of CircuitPython's busio and adafruit_bus_device.
It assumes you have installed smbus2 via `pip install smbus2`.

For the optional reset_pin, it uses OPi.GPIO. Install via `pip install OPi.GPIO` (may vary by Orange Pi model; adjust the GPIO library and mode as needed for your board).

* Original Author(s): ladyada, johnrbnsn
* Conversion for Orange Pi by Grok

Implementation Notes
--------------------

**Hardware:**

* Garmin LIDAR-Lite sensor

**Software and Dependencies:**

* smbus2: https://pypi.org/project/smbus2/
* OPi.GPIO (optional for reset_pin): https://pypi.org/project/OPi.GPIO/

"""

import time
import smbus2 as smbus

# Optional GPIO for reset_pin (adjust import and setmode for your Orange Pi model)
try:
    import OPi.GPIO as GPIO
except ImportError:
    GPIO = None

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_LIDARLite.git"  # Original repo

TYPE_V3 = "V3"
TYPE_V3HP = "V3HP"

CONFIG_DEFAULT = 0
CONFIG_SHORTFAST = 1
CONFIG_DEFAULTFAST = 2
CONFIG_MAXRANGE = 3
CONFIG_HIGHSENSITIVE = 4
CONFIG_LOWSENSITIVE = 5

# Status Registers
# v3
STATUS_BUSY = 0x01
STATUS_REF_OVERFLOW = 0x02
STATUS_SIGNAL_OVERFLOW = 0x04
STATUS_NO_PEAK = 0x08
STATUS_SECOND_RETURN = 0x10
STATUS_HEALTHY = 0x20
STATUS_SYS_ERROR = 0x40

# v3 HP
STATUS_BUSY_V3HP = 0x01
STATUS_SIGNAL_OVERFLOW_V3HP = 0x02

# The various configuration register values, from arduino library
_LIDAR_CONFIGS = (
    (0x80, 0x08, 0x00),  # default
    (0x1D, 0x08, 0x00),  # short range, high speed
    (0x80, 0x00, 0x00),  # default range, higher speed short range
    (0xFF, 0x08, 0x00),  # maximum range
    (0x80, 0x08, 0x80),  # high sensitivity & error
    (0x80, 0x08, 0xB0),  # low sensitivity & error
)

_ADDR_DEFAULT = 0x62
_REG_ACQ_COMMAND = 0x00
_REG_DIST_MEAS_V3 = 0x8F
_REG_DIST_MEAS_V3HP = 0x0F
_REG_SIG_COUNT_VAL = 0x02
_REG_ACQ_CONFIG_REG = 0x04
_REG_THRESHOLD_BYPASS = 0x1C
_REG_STATUS = 0x01
_REG_UNIT_ID_HIGH = 0x16
_REG_UNIT_ID_LOW = 0x17
_REG_SIGNAL_STRENGTH = 0x0E
_REG_HEALTH_STATUS_V3HP = 0x48
_REG_POWER_CONTROL = 0x65
_REG_I2C_CONFIG = 0x1E
_REG_TEST_COMMAND = 0x40
_REG_CORR_DATA = 0x52

_CMD_RESET = 0x00
_CMD_DISTANCENOBIAS = 0x03
_CMD_DISTANCEWITHBIAS = 0x04
_CMD_DISTANCE_V3HP = 0x03


class LIDARLite:
    """
    A driver for the Garmin LIDAR Lite laser distance sensor.

    Initialize the hardware for the LIDAR over I2C. You can pass in an optional reset_pin for when you call reset().
    There are a few common configurations Garmin suggests: CONFIG_DEFAULT, CONFIG_SHORTFAST, CONFIG_DEFAULTFAST,
    CONFIG_MAXRANGE, CONFIG_HIGHSENSITIVE, and CONFIG_LOWSENSITIVE.
    For the I2C address, the default is 0x62 but if you pass a different number in, we'll try to change the address
    so multiple LIDARs can be connected. (Note all but one need to be in reset for this to work!)

    :param int bus_num: The I2C bus number (e.g., 1 for /dev/i2c-1).
    :param int reset_pin: (optional) The GPIO pin number for hardware reset (BOARD mode assumed).
    :param int configuration: (optional) The measurement configuration.
    :param int address: (optional) The I2C address of the device.
    :param str sensor_type: (optional) The sensor type: 'V3' or 'V3HP'.
    """

    def __init__(
        self,
        bus_num=1,
        reset_pin=None,
        configuration=CONFIG_DEFAULT,
        address=_ADDR_DEFAULT,
        sensor_type=TYPE_V3,
    ):
        self.address = address
        self.bus = smbus.SMBus(bus_num)
        self._buf = bytearray(2)
        self._bias_count = 100  # Start with bias on first read
        self._reset_pin = reset_pin
        self._sensor_type = sensor_type

        if self._reset_pin is not None:
            if GPIO is None:
                raise RuntimeError("GPIO library not imported. Cannot use reset_pin.")
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)  # Adjust mode if needed (e.g., GPIO.SUNXI)
            GPIO.setup(self._reset_pin, GPIO.OUT)

        time.sleep(0.5)
        self.reset()
        self.configure(configuration)
        # Note: Address change logic not implemented here; reset others manually if needed.

    def _write_reg(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)
        time.sleep(0.001)

    def _read_reg(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def _read_u16(self, reg):
        data = self.bus.read_i2c_block_data(self.address, reg, 2)
        return (data[0] << 8) | data[1]

    @property
    def status(self):
        return self._read_reg(_REG_STATUS)

    def reset(self):
        """Hardware reset (if pin provided) or software reset. Flushes measurements."""
        if self._reset_pin is not None:
            GPIO.output(self._reset_pin, True)
            GPIO.output(self._reset_pin, False)
            time.sleep(0.01)
            GPIO.output(self._reset_pin, True)
        else:
            try:
                self._write_reg(_REG_ACQ_COMMAND, _CMD_RESET)
            except OSError:
                pass
        time.sleep(1)
        # Flush readings
        for _ in range(100):
            try:
                if self._sensor_type == TYPE_V3:
                    self.read_distance_v3(True)
                else:
                    self.read_distance_v3hp()
            except RuntimeError:
                pass

    def configure(self, config):
        """Set the LIDAR desired style of measurement."""
        if config not in range(6):
            raise ValueError("Configuration out of range")
        settings = _LIDAR_CONFIGS[config]
        self._write_reg(_REG_SIG_COUNT_VAL, settings[0])
        self._write_reg(_REG_ACQ_CONFIG_REG, settings[1])
        self._write_reg(_REG_THRESHOLD_BYPASS, settings[2])

    def read_distance_v3(self, bias=False):
        """Perform a distance reading for V3 with or without bias."""
        if bias:
            self._write_reg(_REG_ACQ_COMMAND, _CMD_DISTANCEWITHBIAS)
        else:
            self._write_reg(_REG_ACQ_COMMAND, _CMD_DISTANCENOBIAS)
        for _ in range(20):
            status = self.status
            if status & STATUS_BUSY == 0:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("Could not obtain measurement")
        dist = self._read_u16(_REG_DIST_MEAS_V3)
        if status & (STATUS_REF_OVERFLOW | STATUS_SIGNAL_OVERFLOW | STATUS_NO_PEAK | STATUS_SECOND_RETURN | STATUS_SYS_ERROR) or not (status & STATUS_HEALTHY):
            raise RuntimeError("Measurement failure")
        return dist

    def read_distance_v3hp(self):
        """Perform a distance reading for V3HP."""
        self._write_reg(_REG_ACQ_COMMAND, _CMD_DISTANCE_V3HP)
        for _ in range(20):
            status = self.status
            if status & STATUS_BUSY_V3HP == 0:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("Could not obtain measurement")
        dist = self._read_u16(_REG_DIST_MEAS_V3HP)
        if status & STATUS_SIGNAL_OVERFLOW_V3HP:
            raise RuntimeError("Measurement failure")
        return dist

    @property
    def distance(self):
        """The measured distance in cm. Will take a bias reading every 100 calls for V3."""
        self._bias_count -= 1
        if self._bias_count < 0:
            self._bias_count = 100
        bias = self._bias_count <= 0
        if self._sensor_type == TYPE_V3:
            return self.read_distance_v3(bias)
        elif self._sensor_type == TYPE_V3HP:
            return self.read_distance_v3hp()
        else:
            raise ValueError("Unknown sensor type")
        return -1

    @property
    def signal_strength(self):
        """The signal strength."""
        return self._read_reg(_REG_SIGNAL_STRENGTH)

    @property
    def unit_id(self):
        """The unit ID."""
        return self._read_u16(_REG_UNIT_ID_HIGH)

    @property
    def health_status(self):
        """The health status (V3HP)."""
        return self._read_reg(_REG_HEALTH_STATUS_V3HP)

    @property
    def power_control(self):
        """The power control (V3HP)."""
        return self._read_reg(_REG_POWER_CONTROL)

    @property
    def i2c_config(self):
        """The I2C config (V3HP)."""
        return self._read_reg(_REG_I2C_CONFIG)

    @property
    def test_command(self):
        """The test command (V3HP)."""
        return self._read_reg(_REG_TEST_COMMAND)

    @property
    def correlation_data(self):
        """The correlation data (V3HP)."""
        return self._read_reg(_REG_CORR_DATA)