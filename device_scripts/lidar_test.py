import time
from smbus import SMBus

i2c_address = 0x62
bus_number = 5

bus = SMBus(bus_number)

def wait_until_ready():
    while True:
        status = bus.read_byte_data(i2c_address, 0x01)
        if not (status & 0x01):
            break
        time.sleep(0.005)

def read_distance():
    bus.write_byte_data(i2c_address, 0x00, 0x04)

    wait_until_ready()

    high = bus.read_byte_data(i2c_address, 0x0f)
    low = bus.read_byte_data(i2c_address, 0x10)

    distance = (high << 8) + low
    
    return distance

while True:
    try:

        dist = read_distance()
        print(f"Distance = {dist}cm", end="\r")
        time.sleep(0.5)
    except KeyboardInterrupt:
        break

