import functools
import itertools
import logging
from z3 import *
from tools import *
from time import time
from itertools import islice, accumulate
import numpy as np

logging.basicConfig(level=logging.DEBUG)

SOLUTION_COUNT = 1  # None for all solutions
WIDTH, HEIGHT = 4, 4
SIZE = WIDTH * HEIGHT
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

CONSTANTS = {
    # (0, 0): False,
    # (WIDTH - 1, HEIGHT - 1): True,
    # (0, HEIGHT - 1): True,
    # (WIDTH - 1, 0): True,
    # (0, 2): 3,
    # (2, 0): 0,
    # (0, 0): 5,
    # (1, 2): 4,
    # (3, 2): 3,
    # (0, 4): 3,
    # (2, 4): 1,
    # (4, 4): 4,
}

s = Solver()
ts = time()

logging.debug("defining value spaces")
t = time()
grid = np.empty((WIDTH, HEIGHT), dtype=ExprRef)
for index in np.ndindex(*grid.shape):
    grid[index] = Bool(f"cell_{'_'.join(str(v) for v in index)}")
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: no 2x2 shaded")
t = time()
# 2x2 shaded cells are not allowed
for x in range(0, WIDTH - 1):
    for y in range(0, HEIGHT - 1):
        s.add(Not(And([grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]])))
logging.debug(f"{time() - t:f} seconds")

logging.debug("constructing: view matrix")
t = time()
right, down = np.array([1, 0]), np.array([0, 1])
cardinals = [right, down, -right, -down]
edge = np.asarray(grid.shape)
view = np.empty(grid.shape, dtype=ExprRef)
for index in np.ndindex(*view.shape):
    start = np.asarray(index)
    visible = []
    for direction in cardinals:
        ray = []
        a = 1
        p = start + a * direction
        while min(p) >= 0 and min(edge - p) > 0:
            ray.append(p)
            a = a + 1
            p = start + a * direction
        if len(ray) == 0:
            continue
        cells = [grid[tuple(p)] for p in ray]
        visible_in_direction = list(accumulate(cells, And))
        visible.extend(visible_in_direction)
    view[index] = Sum([If(cell, 1, 0) for cell in visible])
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: CONSTANTS")
# hard set these coordinates
for key, value in CONSTANTS.items():
    t = time()
    if key < (0, 0) or key >= (WIDTH, HEIGHT):
        raise ValueError(f"Constant key '{key}' is outside of board range")
    if type(value) is bool:
        logging.debug(f"constraining: grid at {key} is {value}")
        s.add(grid[key] == value)
    elif type(value) is int:
        if 0 > value or value > WIDTH + HEIGHT - 2:
            raise ValueError(f"Invalid constant number for {value} in CONSTANTS at key {key}")
        logging.debug(f"constraining: view at {key} is {value}, grid at {key} is {False}")
        s.add(view[key] == value)
        s.add(Not(grid[key]))
    else:
        raise ValueError(f"Invalid constant value type '{type(value)}'")
    logging.debug(f"{time() - t:f} seconds")

    logging.debug("checking sat")
    t = time()
    sat_result = s.check()
    logging.debug(f"{time() - t:f} seconds")

    if sat_result == unsat:
        print("The latest constraint caused an unsat D:")
        exit(1)

logging.debug("constructing: adjacency matrix")
t0 = time()
adjacency = np.empty((SIZE - 2, SIZE, SIZE), dtype=ExprRef)
# the adjacency matrix adjacency[0][i][j] equals 1 when cell#i and cell#j in grid are shaded and connected, otherwise 0
for index in np.ndindex(*adjacency[0].shape):
    i, j = index
    difference = abs(i - j)
    cardinal_neighbors = difference == WIDTH or (
            difference == 1 and not abs(i % WIDTH - j % WIDTH) != 1)
    if cardinal_neighbors:
        adjacency[0][i][j] = And(grid[coordinate(i)], grid[coordinate(j)])
    else:
        adjacency[0][i][j] = False
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
t1 = time()
logging.debug(f"{t1 - t:f} seconds")
logging.debug(f"constructing constraints total time: {t1 - ts:f} seconds")

logging.debug("constructing: ")

logging.debug("finished constraining puzzle rules")
# logging.debug("exporting solver assertions to file:")
# t = time()
# file_name = f"{WIDTH}_{HEIGHT}.smt"
# with open(file_name, "w") as f:
#     f.write(s.to_smt2())
# logging.debug(f"{time() - t:f} seconds")

logging.debug("checking sat")
t = time()
# are we SAT?
sat_result = s.check()
t1 = time()
logging.debug(f"{t1 - t:f} seconds")
logging.debug(f"Total time: {t1 - ts:f} seconds")

if sat_result == unsat:
    print("We are not SAT D:")
    exit(1)

for key, value in CONSTANTS.items():
    if type(value) is bool:
        value = ('' if value else 'un') + 'shaded'
    print(f"{key} will be {value}")

free_terms = [grid[index] for index in np.ndindex(*grid.shape) if index not in CONSTANTS]
s_ = Solver()
s_.assert_exprs(s.assertions())

max_unshaded = SIZE // 2
unshaded_count = Sum([If(grid[c], 0, 1) for c in np.ndindex(*grid.shape)])
s_.add(unshaded_count <= max_unshaded)

for c in np.ndindex(*grid.shape):
    s_.add(view[c] < WIDTH + HEIGHT - 2)

solutions = all_smt(s_, free_terms)
sl = islice(solutions, SOLUTION_COUNT)

t = time()
for i, m in enumerate(sl, start=1):
    logging.debug(f"{time() - t:f} seconds")
    m: ModelRef

    eval_bool_func = np.vectorize(lambda expr: is_true(m.eval(expr, model_completion=True)))
    eval_int_func = np.vectorize(lambda expr: m.eval(expr, model_completion=True).as_long())
    shading = eval_bool_func(grid)
    numbers = eval_int_func(view)
    cell_display_func = np.vectorize(cell_display_l(shading, numbers))
    cells = np.fromfunction(cell_display_func, grid.shape, dtype=int)
    print(f"Solution #{i}:")
    mat_display(cells)
    print()

    # generate puzzles with the given solved board
    def same_number(c):
        return And(fix_term_expr(m, grid[c]), fix_term_expr(m, view[c]))


    number_coordinates = [c for c in np.ndindex(*grid.shape) if c not in CONSTANTS and not shading[c]]
    constraints = [same_number(c) for c in number_coordinates]
    stop = 2
    p = puzzles(s, constraints, free_terms, stop)
    logging.debug(f"iterating puzzles with less than {stop} solutions:")
    t = time()
    count = 0
    min_combs = []
    min_len = len(number_coordinates)
    while comb := next(p, None):
        count += 1
        length = len(comb)
        if length > min_len:
            continue
        if length < min_len:
            min_len = length
            min_combs.clear()
        min_combs.append(comb)
    logging.debug(f"{time() - t:f} seconds")
    print(f"combinations count: {count}")
    print(f"min combination length: {min_len}")
    puzzle_count = 10
    for n, comb in enumerate(islice(min_combs, puzzle_count), start=1):
        print(f"Puzzle #{n}:")
        puzzle = np.empty(grid.shape, dtype=str)
        for c in np.ndindex(*puzzle.shape):
            puzzle[c] = ' '
        for j in comb:
            coordinate = number_coordinates[j]
            value = numbers[coordinate]
            puzzle[coordinate] = str(value)
        mat_display(puzzle)

    print()
    t = time()
if not next(solutions, None):
    print("No more solutions")
