from z3 import *
from mdArray import *
from tools import mat_display, in_range
import numpy as np
from time import time
from math import prod

s = Solver()

N = Int('N')
s.add(N == 2)

box_shape = N, N
box_length = prod(box_shape)

grid_shape = tuple([d * d for d in box_shape])
grid_length = prod(grid_shape)
grid = Array('grid', IntSort(), IntSort())
# s.add(Length(grid) == grid_length)

grid_i = Int('grid_i')
i_in_range = in_range(grid_i, 0, grid_length)

is_digit = in_range(grid[grid_i], 1, box_length + 1)
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
# s.add(ForAll([grid_i, grid_j], Implies(
#     same_row,
#     ij_distinct)))

# col:
same_col = x1 == x2
# s.add(ForAll([grid_i, grid_j], Implies(
#     same_col,
#     ij_distinct)))


def grid_to_box(i):
    return blur_sd(i, grid_shape, box_shape)


box_i, box_j = grid_to_box(grid_i), grid_to_box(grid_j)
same_box = box_i == box_j
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

eval_int = np.vectorize(lambda expr: m.eval(expr, model_completion=True).as_long())
grid_length_v = eval_int(grid_length)
grid_shape_v = eval_int(grid_shape)
board = np.empty(grid_shape_v, dtype=int)
for i in range(grid_length_v):
    coordinates = sd_to_md(i, grid_shape_v)
    board[coordinates] = eval_int(grid[IntVal(i)])
mat_display(board)
