import random

def get_distance():
    """
    Simulates reading from a LIDAR sensor.
    
    In a real application, this function would interface with the hardware.
    For now, it returns a simulated distance.
    """
    # Simulate a distance that is usually safe but occasionally gets closer.
    if random.random() < 0.1: # 10% chance to be close
        return random.uniform(1.0, 2.5)
    else:
        return random.uniform(5.0, 10.0)
