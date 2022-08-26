from z3 import *
from mdArray import *
from tools import mat_display, in_range
import numpy as np
from time import time
from math import prod

s = Solver()

I = IntSort()
N = Int('N')
s.add(N == 2)

box_shape = N, N
box_length = prod(box_shape)

grid_shape = tuple([d * d for d in box_shape])
grid_length = prod(grid_shape)

grid = Array('grid', I, I)

grid_i = Int('grid_i')
s.add(in_range(grid_i, 0, grid_length))

is_digit = in_range(grid[grid_i], 1, grid_length + 1)
s.add(ForAll(grid_i, is_digit))

grid_j = Int('grid_j')
s.add(in_range(grid_j, 0, grid_length))

# grid items are distinct for
ij_distinct = Implies(
    Distinct(grid_i, grid_j),
    Distinct(grid[grid_i], grid[grid_j]))

x1, y1 = sd_to_md(grid_i, grid_shape)
x2, y2 = sd_to_md(grid_j, grid_shape)

# row:
same_row = x1 == x2
s.add(ForAll([grid_i, grid_j], Implies(
    same_row,
    ij_distinct)))

# col:
same_col = y1 == y2
s.add(ForAll([grid_i, grid_j], Implies(
    same_col,
    ij_distinct)))

# box:
# box_i, box_j = Ints('box_i box_j')
# s.add(in_range(box_i, 0, box_length))
# s.add(in_range(box_j, 0, box_length))

box1_x, box1_y = sd_to_md(grid_i, box_shape)
box2_x, box2_y = sd_to_md(grid_j, box_shape)

same_box = And(box1_x == box2_x, box1_y == box2_y)

box_distinct = Implies(
    same_box,
    Distinct()
)

# slice_i = Int('slice_i')
# s.add(in_range(slice_i, 0, grid_shape[))
# tuple([for d in grid_shape])
# s.add(in_range(slice_i, ))
# i, j = Ints('i j')
# i_is_index = in_range(i, 0, NS)
# s.add(i_is_index)
# j_is_index = in_range(j, 0, NS)
# s.add(j_is_index)

# distinct_col = Implies(
#     And(  # x_is_index,
#         i_is_index,
#         j_is_index,
#         Distinct(i, j)),
#     Distinct(grid[0][i], grid[0][j]))
# s.add(ForAll([i, j], distinct_col))

# distinct_row = Implies(
#     And(y_is_index,
#         i_is_index,
#         j_is_index,
#         Distinct(i, j)),
#     Distinct(grid[i][y], grid[j][y]))
# s.add(ForAll([y, i, j], distinct_row))
# xij_are_index = And(x_is_index, i_is_index, j_is_index)
# s.add(ForAll([x, i, j], Implies(xij_are_index, distinct_row)))

# patterns=[x_is_index, i_is_index, j_is_index]))

# for rows
# row_pair = (i, y), (j, y)
# s.add(ForAll([y, i, j], same_position(*row_pair) == same_value(grid, *row_pair)))


# for boxes
# box_x, box_y = Ints('box_x box_y')


# box_pair = (box_x*N+x, box_y*N+y), ()
# s.add(box_x == x / N)
# s.add(box_y == y / N)
# inside_x, inside_y = Ints('inside_x inside_y')
# s.add(inside_x == x % N)
# s.add(inside_y == y % N)

# s.add(ForAll([i, j], same_position())))

t = time()
check_sat_result = s.check()
print(f"{time() - t} seconds")
if check_sat_result == unsat:
    print("unsat")
exit(1)

m = s.model()
n = m.eval(N, model_completion=True).as_long()
ns = n ** 2
board = np.empty((ns, ns), dtype=int)
for index in np.ndindex(*board.grid_shape):
    x, y = index
board[index] = m.eval(grid[x][y], model_completion=True).as_long()
mat_display(board)
