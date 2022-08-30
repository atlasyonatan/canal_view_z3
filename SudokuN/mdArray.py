import math

from z3 import Select, Store, QuantifierRef, Solver, Z3_OP_MUL, IntVal, Sum, ExprRef
from operator import mul
from itertools import accumulate

ExprRef.__floordiv__ = lambda self, other: self / other


def sd_to_md(i, shape):
    coordinates = [i]
    for length in shape[:-1]:
        oneD = coordinates.pop()
        twoD = oneD % length, oneD // length
        coordinates.extend(twoD)
    return tuple(coordinates)


def md_to_sd(coordinates, shape):
    dmult = list(accumulate(list(shape), mul, initial=1))
    return sum(dmult[i] * coordinates[i] for i in range(len(shape)))


def blur_md(md, blur):
    return tuple(md[i] // blur[i] for i in range(len(md)))


def blur_sd(i, shape, blur):
    coordinates = sd_to_md(i, shape)
    new_coordinates = blur_md(coordinates, blur)
    new_shape = blur_md(shape, blur)
    new_i = md_to_sd(new_coordinates, new_shape)
    return new_i

# def md_eval(s: Solver, m, arr, *index):
#     f = m.eval(arr, model_completion=True)
#     for i in index:
#         if isinstance(f, QuantifierRef):
#             # s.push()
#             s.add(f)
#             s.check()
#             m = s.model()
#             # print(s.check())
#             # s.pop()
#         arr = f(i)
#         f = m.eval(arr, model_completion=True)
#     return f


# def md_select(arr, index):
#     for i in index:
#         arr = Select(arr, i)
#     return arr
#
#
# def md_store(arr, index, value):
#     degree = len(index)
#     arr_refs = [arr]
#     for i in range(degree - 1):
#         arr = Select(arr_refs[i], index[i])
#         arr_refs.append(arr)
#     for i in range(degree - 1, -1, -1):
#         arr = arr_refs.pop()
#         value = Store(arr, index[i], value)
#     return value
