class Properties:
    def __init__(self, props: dict):
        self.__dict__.update(props)


def centroid(x1: int, y1: int, x2: int, y2: int):
    xc = x1 + ((x2 - x1) >> 1)
    yc = y1 + ((y2 - y1) >> 1)

    return xc, yc
