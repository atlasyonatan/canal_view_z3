from z3 import *
from mdArray import *
from tools import mat_display, in_range
import numpy as np
from time import time

s = Solver()

I = IntSort()
N = Int('N')
NS = N * N
s.add(N == 2)

grid = Array('grid', I, ArraySort(I, I))

# grid items are [1..NS]
x, y = Ints('x y')
x_is_index = in_range(x, 0, NS)
s.add(x_is_index)
y_is_index = in_range(y, 0, NS)
s.add(y_is_index)

is_digit = in_range(grid[x][y], 1, NS + 1)
s.add(ForAll([x, y], is_digit))

# grid items are distinct
i, j = Ints('i j')
i_is_index = in_range(i, 0, NS)
# s.add(i_is_index)
j_is_index = in_range(j, 0, NS)
# s.add(j_is_index)

distinct_col = Implies(
    And(#x_is_index,
        i_is_index,
        j_is_index,
        Distinct(i, j)),
    Distinct(grid[0][i], grid[0][j]))
s.add(ForAll([i, j], distinct_col))

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
for index in np.ndindex(*board.shape):
    x, y = index
    board[index] = m.eval(grid[x][y], model_completion=True).as_long()
mat_display(board)
