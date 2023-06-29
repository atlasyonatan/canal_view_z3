from operator import mul
from itertools import accumulate, starmap
from functools import partial, reduce
from z3 import ExprRef

compose = lambda *fs: reduce(lambda f, g: lambda *a, **kw: f(g(*a, **kw)), fs)
ExprRef.__floordiv__ = lambda self, other: self / other

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
