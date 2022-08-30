from z3 import *
from mdArray import *
from tools import mat_display, in_range
import numpy as np
from time import time
from math import prod

s = Solver()
# print(s.help())
# s.set('mbqi.max_iterations', 10)
# s.set("mbqi.trace", True)
# s.set("mbqi", False)
s.set("mbqi.trace", True)
# s.set(auto_config=False, mbqi=False)

N = Int('N')
s.add(N == 1)

box_shape = N, N
box_length = prod(box_shape)

grid_shape = tuple([d * d for d in box_shape])
grid_length = prod(grid_shape)
grid = Array('grid', IntSort(), IntSort())
# s.add(Length(grid) == grid_length)

i = Int('i')
i_in_range = in_range(i, 0, grid_length)
s.add(i_in_range)

is_digit = in_range(grid[i], 1, box_length + 1)
# is_zero = grid[grid_i] == 0
s.add(ForAll(i, is_digit, patterns=[]))
# s.add(ForAll(i, is_digit))
# s.add(ForAll(grid_i, is_digit, patterns=[grid[grid_i]]))
# s.add(ForAll([grid_i], If(i_in_range, is_digit, is_zero)))

j = Int('grid_j')
j_in_range = in_range(j, 0, grid_length)
s.add(j_in_range)

# grid items are distinct for
ij_distinct = Implies(
    Distinct(i, j),
    Distinct(grid[i], grid[j]))
# ij_distinct = Implies(
#     And(i_in_range,
#         j_in_range,
#         Distinct(grid_i, grid_j)),
#     Distinct(grid[grid_i], grid[grid_j]))

x1, y1 = sd_to_md(i, grid_shape)
x2, y2 = sd_to_md(j, grid_shape)

# row:
same_row = y1 == y2
s.add(ForAll([i, j], Implies(
    same_row,
    ij_distinct)))

# col:
same_col = x1 == x2


s.add(ForAll([i, j], Implies(
    same_col,
    ij_distinct)))


def grid_to_box(i):
    return blur_sd(i, grid_shape, box_shape)


box_i, box_j = grid_to_box(i), grid_to_box(j)
same_box = box_i == box_j
s.add(ForAll([i, j], Implies(
    same_box,
    ij_distinct)))

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
for index in range(grid_length_v):
    coordinates = sd_to_md(index, grid_shape_v)
    board[coordinates] = eval_int(grid[IntVal(index)])
mat_display(board)
