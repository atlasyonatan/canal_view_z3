from z3 import *
from mdArray import *
from tools import mat_display
import numpy as np
from time import time

s = Solver()

I = IntSort()
N = Int('N')
NS = N * N
s.add(N == 3)


def in_range(expr, start, stop):
    return And(expr >= start, expr < stop)


def is_index(expr, length):
    return in_range(expr, 0, length)


grid = Array('grid', I, ArraySort(I, I))

# grid items are [1..NS]
x, y = Ints('x y')
is_digit = in_range(grid[x][y], 1, NS + 1)
s.add(ForAll([x, y], is_digit))

# grid items are distinct
i, j = Ints('i j')
# for rows
row_pair = (0, i), (0, j)
s.add(ForAll([i, j], same_position(*row_pair) == same_value(grid, *row_pair)))

# for columns
col_pair = (i, 0), (j, 0)
s.add(ForAll([i, j], same_position(*col_pair) == same_value(grid, *col_pair)))

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
    board[index] = md_eval(m, grid, *index).as_long()
mat_display(board)
