import logging
from z3 import *
from tools import *
from time import time
from itertools import islice
import numpy as np

# logging.basicConfig(level=logging.DEBUG)

SOLUTION_COUNT = 1
HEIGHT, WIDTH = 3, 3
SIZE = WIDTH * HEIGHT
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

CONSTANTS = {
    (0, 0): True,
    (1, 0): True,
    (2, 0): True,
}

s = Solver()

logging.debug("defining value spaces")
t = time()
grid = np.empty((WIDTH, HEIGHT), dtype=ExprRef)
for index in np.ndindex(*grid.shape):
    grid[index] = Bool(f"cell_{'_'.join(str(v) for v in index)}")
# grid = [[Bool(f'cell_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH)]
# adjacency = [[Bool(f'adjacency_{i}_{j}') for j in range(SIZE)] for i in range(SIZE)]
# numbers = [ [Int(f'number_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH) ]
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: constants")
# hard set these coordinates
for key, value in CONSTANTS.items():
    s.push()
    t = time()
    constraint = f"'{key} == {value}'"
    logging.debug(f"constraining: {constraint}")
    s.add(grid[key] == value)
    logging.debug(f"{time() - t:f} seconds")

    logging.debug("checking sat")
    t = time()
    # are we SAT?
    sat_result = s.check()
    logging.debug(f"{time() - t:f} seconds")

    if sat_result == unsat:
        print(f"This constraint: {constraint} causes an unsat D:")
        exit(1)

logging.debug("constraining: no 2x2 shaded")
t = time()
# 2x2 shaded cells are not allowed
for x in range(0, WIDTH - 1):
    for y in range(0, HEIGHT - 1):
        s.add(Not(And([grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]])))
logging.debug(f"{time() - t:f} seconds")

adjacency = np.empty((SIZE - 2, SIZE, SIZE), dtype=ExprRef)

logging.debug("constructing: adjacency matrix")
t0 = time()
# the adjacency matrix adjacency[0][i][j] equals 1 when cell#i and cell#j in grid are shaded and connected, otherwise 0
for index in np.ndindex(*adjacency[0].shape):
    i, j = index
    d = abs(i - j)
    connected = d == 1 or d == WIDTH or d == 0
    if connected:
        mark = And(grid[coordinate(i)], grid[coordinate(j)])
    else:
        mark = BoolVal(False)
    # adjacency_k[0][i][j] = If(mark, 1, 0)
    adjacency[0][i][j] = mark
logging.debug(f"{time() - t:f} seconds")

logging.debug("constructing: adjacency_k")
t0 = time()
# powers of The Adjacency Matrix
for k in range(1, adjacency.shape[0]):
    adjacency[k] = np.dot(adjacency[0], adjacency[k - 1])  # dot product
logging.debug(f"{time() - t:f} seconds")

logging.debug("constructing: sum_adjacency_k")
t0 = time()
# matrix for the sum of all adjacency^k
adjacency_k_sum = np.ndarray.sum(adjacency, axis=0)
logging.debug(f"{time() - t:f} seconds")

# logging.debug("constraining: sum of adjacency^k is nonzero for shaded cell pairs")
# t = time()
# # constrain the sum of adjacency^k for k in [1..SIZE-1], is positive for all shaded cells
# for i in range(SIZE):
#     for j in range(SIZE):
#         x1, y1 = coordinate(i)
#         x2, y2 = coordinate(j)
#         nonzero = adjacency_k_sum[i][j]
#         shaded = And(grid[x1][y1], grid[x2][y2])
#         s.add(shaded == nonzero)
# logging.debug(f"{time() - t:f} seconds")
#
logging.debug("finished constraining puzzle rules")

logging.debug("checking sat")
t = time()
# are we SAT?
sat_result = s.check()
logging.debug(f"{time() - t:f} seconds")

if sat_result == unsat:
    print("We are not SAT D:")
    exit(1)

free_terms = [grid[index] for index in np.ndindex(*grid.shape) if index not in CONSTANTS]
# for index in np.ndindex(*grid.shape):
#     if index not in CONSTANTS:
#         free_terms.append(grid[index])
solutions = all_smt(s, free_terms)
s = islice(solutions, SOLUTION_COUNT)
for i, m in enumerate(s, start=1):
    m: ModelRef
    print(f"Solution #{i}:")


    def display_bool(expr):
        v = m.eval(expr, model_completion=True)
        return '#' if v else ' '


    mat_display(grid, display_bool)


    def display_int(expr):
        v = m.eval(expr, model_completion=True)
        return '1' if v else '0'

    count = 2
    for k, adjacency_k in islice(enumerate(adjacency), count):
        print(f"adjacency^{k+1}:")
        mat_display(adjacency_k, display_bool)

    print(f"adjacency_k_sum:")
    mat_display(adjacency_k_sum, display_bool)
    print()
