import logging
from z3 import *
from tools import *
from time import time
from itertools import islice
import numpy as np

logging.basicConfig(level=logging.DEBUG)

SOLUTION_COUNT = 4  # None for all solutions
WIDTH, HEIGHT = 4, 4
SIZE = WIDTH * HEIGHT
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

CONSTANTS = {
    (0, 0): True,
    (WIDTH-1, HEIGHT-1): True,
    (0, HEIGHT-1): True,
    (WIDTH-1, 0): True,
}
#
# for x in range(WIDTH):
#     CONSTANTS[(x, HEIGHT//2)] = False
#
# CONSTANTS[(WIDTH//2, HEIGHT//2)] = True

s = Solver()

logging.debug("defining value spaces")
t = time()
grid = np.empty((WIDTH, HEIGHT), dtype=ExprRef)
for index in np.ndindex(*grid.shape):
    grid[index] = Bool(f"cell_{'_'.join(str(v) for v in index)}")
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: constants")
# hard set these coordinates
for key, value in CONSTANTS.items():
    t = time()
    constraint = f"'{key} == {value}'"
    logging.debug(f"constraining: {constraint}")
    s.add(grid[key] == value)
    logging.debug(f"{time() - t:f} seconds")

    logging.debug("checking sat")
    t = time()
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
    cardinal_neighbors = d == 0 or d == WIDTH or (d == 1 and not abs(i % WIDTH - j % WIDTH) != 1)
    if cardinal_neighbors:
        adjacency[0][i][j] = And(grid[coordinate(i)], grid[coordinate(j)])
    else:
        adjacency[0][i][j] = BoolVal(False)
logging.debug(f"{time() - t:f} seconds")

logging.debug("constructing: adjacency_k")
t0 = time()

# powers of The Adjacency Matrix
for k in range(1, adjacency.shape[0]):
    mat_mul = z3_bool_mat_mul(adjacency[0], adjacency[k - 1])
    for index in np.ndindex(*adjacency[k].shape):
        adjacency[k][index] = mat_mul(*index)
logging.debug(f"{time() - t:f} seconds")

logging.debug("constructing: sum_adjacency_k")
t = time()

# matrix for the sum of all adjacency^k
mat_sum = z3_bool_mat_sum(adjacency)
adjacency_k_sum = np.empty(adjacency[0].shape, dtype=ExprRef)
for index in np.ndindex(*adjacency_k_sum.shape):
    adjacency_k_sum[index] = mat_sum(*index)
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: sum of adjacency^k is nonzero for shaded cell pairs")
t = time()
# constrain the sum of adjacency^k for k in [1..SIZE-1], is positive for all shaded cells
for index in np.ndindex(*adjacency_k_sum.shape):
    i, j = index
    nonzero = adjacency_k_sum[index]
    shaded = And(grid[coordinate(i)], grid[coordinate(j)])
    s.add(shaded == nonzero)
logging.debug(f"{time() - t:f} seconds")

logging.debug("finished constraining puzzle rules")

logging.debug("checking sat")
t = time()
# are we SAT?
sat_result = s.check()
logging.debug(f"{time() - t:f} seconds")

if sat_result == unsat:
    print("We are not SAT D:")
    exit(1)

for key, value in CONSTANTS.items():
    print(f"{key} will be {'' if value else 'un'}shaded")

free_terms = [grid[index] for index in np.ndindex(*grid.shape) if index not in CONSTANTS]
solutions = all_smt(s, free_terms)
s = islice(solutions, SOLUTION_COUNT)
t = time()
for i, m in enumerate(s, start=1):
    logging.debug(f"{time() - t:f} seconds")
    m: ModelRef
    evaluate = np.vectorize(lambda expr: is_true(m.eval(expr, model_completion=True)))
    eval_grid = evaluate(grid)
    print(f"Solution #{i}:")
    mat_display(eval_grid, bool_display)

    # count = 1
    # for k, adjacency_k in islice(enumerate(adjacency), count):
    #     eval_adjacency_k = evaluate(adjacency_k)
    #     print(f"adjacency^{k + 1}:")
    #     mat_display(eval_adjacency_k, bool_display)

    # eval_adjacency_k_sum = evaluate(adjacency_k_sum)
    # print(f"adjacency_k_sum:")
    # mat_display(eval_adjacency_k_sum, bool_display)

    print()
    t = time()
if not next(solutions, None):
    print("No more solutions")
