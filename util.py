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
