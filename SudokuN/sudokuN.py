from z3 import *
from mdArray import *
from tools import mat_display, in_range
import numpy as np
from time import time
from math import prod

s = Solver()

N = Int('N')
s.add(N == 1)

box_shape = N, N
box_length = prod(box_shape)

grid_shape = tuple([d * d for d in box_shape])
grid_length = prod(grid_shape)

grid = Array('grid', IntSort(), IntSort())

grid_i = Int('grid_i')
i_in_range = in_range(grid_i, 0, grid_length)

is_digit = in_range(grid[grid_i], 1, grid_length + 1)
is_zero = grid[grid_i] == 0
s.add(ForAll([grid_i], If(i_in_range, is_digit, is_zero)))

grid_j = Int('grid_j')
j_in_range = in_range(grid_j, 0, grid_length)

# grid items are distinct for
ij_distinct = Implies(
    And(i_in_range,
        j_in_range,
        Distinct(grid_i, grid_j)),
    Distinct(grid[grid_i], grid[grid_j]))

x1, y1 = sd_to_md(grid_i, grid_shape)
x2, y2 = sd_to_md(grid_j, grid_shape)

# row:
same_row = y1 == y2
s.add(ForAll([grid_i, grid_j], Implies(
    same_row,
    ij_distinct)))

# col:
# same_col = x1 == x2
# s.add(ForAll([grid_i, grid_j], Implies(
#     same_col,
#     ij_distinct)))
#
#
# def grid_to_box(i):
#     return blur_sd(i, grid_shape, box_shape)
#
#
# box_i, box_j = grid_to_box(grid_i), grid_to_box(grid_j)
# same_box = box_i == box_j
# s.add(ForAll([grid_i, grid_j], Implies(
#     same_box,
#     ij_distinct)))

t = time()
check_sat_result = s.check()
print(f"{time() - t} seconds")
if check_sat_result == unsat:
    print("unsat")
    exit(1)

m = s.model()
grid_shape_v = np.vectorize(lambda expr: m.eval(expr, model_completion=True).as_long())(grid_shape)
# grid_v = evaluate(grid)
# grid_length_v = evaluate(grid_length)
# grid_v = m.eval(grid, model_completion=True)
# s.add(grid_v)
index = Int('index')
value = m.eval(grid[index])

board = np.empty(grid_shape_v, dtype=int)
# for coordinates in np.ndindex(*grid_shape_v):
#     i = md_to_sd(coordinates, grid_shape_v)
#     solver = Solver()
#     solver.add(index == IntVal(i))
#     if solver.check() == unsat:
#         raise ValueError(f"no solution for {index=}={i} and {value=}=\n{value}")
#     model = solver.model()
#     v = model.evaluate(value).as_long()
#     board[coordinates] = v
mat_display(board)
