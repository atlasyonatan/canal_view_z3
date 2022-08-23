#!/usr/bin/env -S python
import logging
from z3 import *
from tools import *
from time import time
from itertools import islice
import numpy as np

logging.basicConfig(level=logging.DEBUG)

HEIGHT, WIDTH = 5, 5
SIZE = WIDTH * HEIGHT
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")

s = Solver()

logging.debug("defining value spaces")
t0 = time()
grid = [[Bool(f'cell_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH)]
# adjacency = [[Bool(f'adjacency_{i}_{j}') for j in range(SIZE)] for i in range(SIZE)]
# numbers = [ [Int(f'number_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH) ]
t1 = time()
logging.debug(f"{t1 - t0:f} seconds")

logging.debug("constraining: no 2x2 shaded")
t0 = time()
# 2x2 shaded cells are not allowed
for x in range(0, WIDTH - 1):
    for y in range(0, HEIGHT - 1):
        s.add(Not(And([grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]])))
t1 = time()
logging.debug(f"{t1 - t0:f} seconds")

# shape = (SIZE - 2, SIZE, SIZE)
# adjacency_k = np.empty(shape, dtype=ExprRef)
# for index in np.ndindex(*adjacency_k.shape):
#     adjacency_k[index] = Bool(f"adjacency_{'_'.join(str(v) for v in index)}")

# logging.debug("constructing: adjacency_k")
# t0 = time()
# # The Adjacency Matrix (tam) defined to be 1 when cells i and j in grid are shaded and connected, otherwise 0
# for x in range(WIDTH):
#     for y in range(HEIGHT - 1):
#         x1, y1 = x, y
#         x2, y2 = x, y + 1
#         i = cell_number(x1, y1)
#         j = cell_number(x2, y2)
#         both_shaded = And(grid[x][y], grid[x][y + 1])
#         adjacency_k[0][i][j] = both_shaded  # z3bool_to_int(both_shaded)
# for x in range(WIDTH - 1):
#     for y in range(HEIGHT):
#         i = cell_number(x, y)
#         j = cell_number(x + 1, y)
#         both_shaded = And(grid[x][y], grid[x + 1][y])
#         adjacency_k[0][i][j] = both_shaded  # z3bool_to_int(both_shaded)
# t1 = time()
# logging.debug(f"{t1 - t0:f} seconds")

# logging.debug("constructing: adjacency_k")
# t0 = time()
# # powers of The Adjacency Matrix
# for k in range(1, adjacency_k.shape[0]):
#     adjacency_k[k] = adjacency_k[0] * adjacency_k[k - 1]  # dot product
# t1 = time()
# logging.debug(f"{t1 - t0:f} seconds")

# logging.debug("constructing: sum_adjacency_k")
# t0 = time()
# # matrix for the sum of all adjacency^k
# adjacency_k_sum = np.ndarray.sum(adjacency_k, axis=0)
# t1 = time()
# logging.debug(f"{t1 - t0:f} seconds")

# logging.debug("constraining: sum of adjacency^k is nonzero for shaded cell pairs")
# t0 = time()
# # constrain the sum of adjacency^k for k in [1..SIZE-1], is positive for all shaded cells
# for i in range(SIZE):
#     for j in range(SIZE):
#         x1, y1 = coordinate(i)
#         x2, y2 = coordinate(j)
#         shaded = And(grid[x1][y1], grid[x2][y2])
#         s.add(shaded == adjacency_k_sum[i][j])
# t1 = time()
# logging.debug(f"{t1 - t0:f} seconds")

# logging.debug("constraining: constants")
# t0 = time()
#
# for c in constant:
#     print(c, " will be shaded")
#     s.add(grid[c[0]][c[1]])
# t1 = time()
# logging.debug(f"{t1 - t0:f} seconds")

logging.debug("checking sat")
t0 = time()
# are we SAT?
sat_result = s.check()
t1 = time()
logging.debug(f"{t1 - t0:f} seconds")

if s.check() == unsat:
    print("We are not SAT D:")
    exit(1)

# hard set these coordinates
constant_coordinates = [(0, 0), (2, 2), (0, 4)]
for x, y in constant_coordinates:
    s.add(grid[x][y])


# free_terms = [[grid[x][y] for x in range(WIDTH) if (x, y) not in constant_coordinates] for y in range(HEIGHT)]


def cell_display(b):
    return '#' if b else ' '


solutions = all_smt(s, [])
for i, m in enumerate(islice(solutions, 4), start=1):
    m: ModelRef


    def eval_cell(x, y):
        r = grid[x][y]
        v = m.eval(r)
        return v


    print(f"Sat #{i}:")
    mat_display(grid, lambda x, y: cell_display(eval_cell(x, y)))
    print()
# print(m)
