import time


def load_classes(file_path):
    with open(file_path, "r") as fp:
        names = [x for x in fp.read().split("\n") if x]
    return names


def color_map(n):
    def bit_get(x, i):
        return x & (1 << i)

    cmap = []
    for i in range(n):
        r = g = b = 0
        for j in range(7, -1, -1):
            r |= bit_get(i, 0) << j
            g |= bit_get(i, 1) << j
            b |= bit_get(i, 2) << j
            i >>= 3

        cmap.append((r, g, b))
    return cmap


class DurationTimer:
    def __init__(self):
        self.start, self.end = None, None

    @property
    def duration(self):
        return self.end - self.start

    def __enter__(self):
        self.start, self.end = time.time(), None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        return False
