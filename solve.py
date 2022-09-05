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
WIDTH, HEIGHT = 5, 5
SIZE = WIDTH * HEIGHT
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

CONSTANTS = {
    # (0, 0): False,
    # (WIDTH - 1, HEIGHT - 1): True,
    # (0, HEIGHT - 1): True,
    # (WIDTH - 1, 0): True,
    # (1, 1): 3,
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
    cardinal_neighbors = difference == 0 or difference == WIDTH or (
            difference == 1 and not abs(i % WIDTH - j % WIDTH) != 1)
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
t1 = time()
logging.debug(f"{t1 - t:f} seconds")

logging.debug("constructing: ")

logging.debug("finished constraining puzzle rules")
logging.debug(f"constructing constraints total time: {t1 - ts:f} seconds")

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
solutions = all_smt(s, free_terms)
# for i, m in enumerate(solutions, start=1):
#     print(i)
# exit(0)
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


    def same_number(index):
        return And(fix_term(m, grid[index]), fix_term(m, view[index]))


    # number_coordinates = [c for c in np.ndindex(*grid.shape) if not shading[c]]
    # same_number_terms = [same_number(c) for c in number_coordinates]
    # free_coordinates =
    not_constant = [c for c in np.ndindex(*grid.shape) if c not in CONSTANTS]
    shaded = [c for c in not_constant if shading[c]]
    optional_numbers_coordinates = [c for c in not_constant if not shading[c]]
    threshold = 1


    def redundant(start: int = 0):
        for i in range(start, len(optional_numbers_coordinates)):
            s.push()
            is_same = same_number(optional_numbers_coordinates[i])
            s.add(Not(is_same))

            constant_numbers = [optional_numbers_coordinates[j] for j in
                                range(start + 1, len(optional_numbers_coordinates))]
            for c in constant_numbers:
                s.add(same_number(c))

            # redundant_so_far = [optional_numbers_coordinates[j] for j in range(start + 1)]
            # free_coordinates = redundant_so_far + shaded
            # free_terms = [grid[c] for c in free_coordinates]
            # solutions = all_smt(s, free_terms)
            # count = 0
            # while count <= threshold:
            #     solution = next(solutions, None)
            #     if solution is None:
            #         break
            #     count += 1
            # if 0 <= count <= threshold:
            # return count > n
            # if not yields_above(, threshold):
            check = s.check()
            s.pop()
            if check == unsat:
                rec = redundant(i + 1)
                if red := next(rec, None):
                    yield [i] + red
                    yield from ([i] + red for red in rec)
                else:
                    yield [i]
                # for red in :
                #     had_children
                #     yield
                # while red := next(rec, None):
                #     yield [i] + red
                #
                # yield from ([i] + red for red in redundant(i + 1))
                #
                # yield [i]
                # s.push()
                # s.add(is_same)

                # s.pop()


    # print("possible numbers:")
    # print('\n'.join(str(t) for t in enumerate(optional_numbers_coordinates)))
    logging.debug("iterating redundant combinations:")
    t = time()
    r = redundant()
    count = 0
    max_combs = []
    max_len = 0
    while comb := next(r, None):
        count += 1
        if len(comb) < max_len:
            continue
        if len(comb) > max_len:
            max_len = len(comb)
            max_combs.clear()
        max_combs.append(comb)
    logging.debug(f"{time() - t:f} seconds")

    print(f"redundant combinations count: {count}")
    print(f"max redundant length: {max_len}")
    # print(f"longest redundant combinations:")
    # for comb in max_combs:
    #     print(comb)
    # redundant_combinations = red
    puzzle_count = 10
    for n, comb in enumerate(islice(max_combs, puzzle_count), start=1):
        print(f"Puzzle #{n}:")
        puzzle = np.empty(grid.shape, dtype=str)
        for c in np.ndindex(*puzzle.shape):
            puzzle[c] = ' '
        comb = set(comb)
        required = [j for j in range(len(optional_numbers_coordinates)) if j not in comb]
        for j in required:
            coordinate = optional_numbers_coordinates[j]
            value = numbers[coordinate]
            puzzle[coordinate] = str(value)
        mat_display(puzzle)

    print()
    t = time()
if not next(solutions, None):
    print("No more solutions")
