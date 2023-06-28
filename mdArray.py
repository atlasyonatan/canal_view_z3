from operator import mul
from itertools import accumulate, starmap


def sd_to_md(shape):
    def f(i):
        coordinates = [i]
        for length in shape[:-1]:
            oneD = coordinates.pop()
            twoD = oneD % length, oneD // length
            coordinates.extend(twoD)
        return tuple(coordinates)

    return f


def md_to_sd(shape):
    return lambda *coordinates: sum(
        starmap(mul, zip(accumulate(shape, mul, initial=1), coordinates))
    )
