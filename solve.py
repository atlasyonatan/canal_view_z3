import logging
from z3 import *
from tools import *
from time import time
from itertools import islice, accumulate
import numpy as np

logging.basicConfig(level=logging.DEBUG)

SOLUTION_COUNT = 20  # None for all solutions
WIDTH, HEIGHT = 3, 3
SIZE = WIDTH * HEIGHT
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

CONSTANTS = {
    # (0, 0): False,
    # (WIDTH - 1, HEIGHT - 1): True,
    # (0, HEIGHT - 1): True,
    # (WIDTH - 1, 0): True,
    # (4, 1): 6,
    # (2, 2): 5,
    # (1, 3): 3,
    # (3, 4): 3,
    # (0, 4): 3,
    # (2, 4): 1,
    # (4, 4): 4,
}

s = Solver()
ts = time()

logging.debug("defining value spaces")
t = time()

Cell, CellConsts = EnumSort("cell", [str(i) for i in range(SIZE)])


shading = Function("shading", Cell, BoolSort())

grid = np.empty((WIDTH, HEIGHT), dtype=ExprRef)
for index in range(SIZE):
    x, y = coordinate(index)
    grid[x][y] = shading(CellConsts[index])
    # shading(CellConsts[index]) = Bool(f"cell_{x}_{y}")
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: no 2x2 shaded")
t = time()
# 2x2 shaded cells are not allowed
# offsets_2by2 = [(0,0), (0,1), (1,0), (1,1)]
for x in range(0, WIDTH - 1):
    for y in range(0, HEIGHT - 1):
        # cell_numbers = [cell_number(x+dx,y+dy) for (dx,dy) in offsets_2by2]
        # cells = [shading(CellConsts(i)) for i in cell_numbers]
        # s.add(Not(And(cells)))
        # cell_right = (x,y+1)
        # cell_down = (x+1,y)
        # cell_down_right = (x+1,y+1)
        s.add(
            Not(And([grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]]))
        )
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
            raise ValueError(
                f"Invalid constant number for {value} in CONSTANTS at key {key}"
            )
        logging.debug(
            f"constraining: view at {key} is {value}, grid at {key} is {False}"
        )
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

# logging.debug("constructing: adjacency matrix")
# t0 = time()
# adjacency = np.empty((SIZE - 2, SIZE, SIZE), dtype=ExprRef)
# # the adjacency matrix adjacency[0][i][j] equals 1 when cell#i and cell#j in grid are shaded and connected, otherwise 0
# for index in np.ndindex(*adjacency[0].shape):
#     i, j = index
#     difference = abs(i - j)
#     cardinal_neighbors = difference == WIDTH or (
#             difference == 1 and not abs(i % WIDTH - j % WIDTH) != 1)
#     if cardinal_neighbors:
#         adjacency[0][i][j] = And(grid[coordinate(i)], grid[coordinate(j)])
#     else:
#         adjacency[0][i][j] = False
# logging.debug(f"{time() - t:f} seconds")

# logging.debug("constructing: adjacency_k")
# t0 = time()

# # powers of The Adjacency Matrix
# for k in range(1, adjacency.shape[0]):
#     mat_mul = z3_bool_mat_mul(adjacency[0], adjacency[k - 1])
#     for index in np.ndindex(*adjacency[k].shape):
#         adjacency[k][index] = mat_mul(*index)
# logging.debug(f"{time() - t:f} seconds")

logging.debug("constructing: shaded path")
t = time()

# Cell, CellConsts = EnumSort("cell", [str(i) for i in range(SIZE)])

are_neighbors = Function("are_neighbors", Cell, Cell, BoolSort())

# for i, j in np.ndindex((SIZE, SIZE)):
#     x1, y1 = coordinate(i)
#     x2, y2 = coordinate(j)
#     neighbors = (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1)
#     s.add(are_neighbors(grid[x1][y1], grid[x2][y2]) == neighbors)


for i, j in np.ndindex((SIZE, SIZE)):
    x1, y1 = coordinate(i)
    x2, y2 = coordinate(j)
    neighbors = (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1)
    s.add(are_neighbors(CellConsts[i], CellConsts[j]) == neighbors)

are_shaded_neighbors = Function("are_shaded_neighbors", Cell, Cell, BoolSort())

i_q, j_q = Consts("i_q j_q", Cell)
both_shaded_q = And(shading(i_q), shading(j_q))
s.add(
    ForAll(
        [i_q, j_q],
        are_shaded_neighbors(i_q, j_q) == And(are_neighbors(i_q, j_q), both_shaded_q),
    )
)


# for i, j in np.ndindex((SIZE, SIZE)):
#     # print(f"i:{i}, j:{j}")
#     x1, y1 = coordinate(i)
#     x2, y2 = coordinate(j)
#     are_neighbors = (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1)
#     both_shaded = And(grid[coordinate(i)], grid[coordinate(j)])
#     s.add(
#         are_shaded_neighbors(CellConsts[i], CellConsts[j])
#         == And(both_shaded, are_neighbors)
#     )

shaded_path = TransitiveClosure(are_shaded_neighbors)
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: shaded path")
t = time()

# s.add(ForAll([i_q, j_q], shaded_path(i_q, j_q) == both_shaded_q))

# for i, j in np.ndindex((SIZE, SIZE)):
#     if i != j:
#         both_shaded = And(grid[coordinate(i)], grid[coordinate(j)])
#         constraint = shaded_path(CellConsts[i], CellConsts[j]) == both_shaded
#         # print("adding: " + constraint.__repr__())
#         s.add(constraint)
#         # print(s.check())

logging.debug(f"{time() - t:f} seconds")

# # matrix for the sum of all adjacency^k
# mat_sum = z3_bool_mat_sum(adjacency)
# adjacency_k_sum = np.empty(adjacency[0].shape, dtype=ExprRef)
# for index in np.ndindex(*adjacency_k_sum.shape):
#     adjacency_k_sum[index] = mat_sum(*index)
# logging.debug(f"{time() - t:f} seconds")

# logging.debug("constraining: sum of adjacency^k is nonzero for shaded cell pairs")
# t = time()
# # constrain the sum of adjacency^k for k in [1..SIZE-1], is positive for all shaded cells
# for index in np.ndindex(*adjacency_k_sum.shape):
#     i, j = index
#     nonzero = adjacency_k_sum[index]
#     shaded = And(grid[coordinate(i)], grid[coordinate(j)])
#     s.add(shaded == nonzero)
t1 = time()
# logging.debug(f"{t1 - t:f} seconds")
logging.debug(f"constructing constraints total time: {t1 - ts:f} seconds")

logging.debug("finished constraining puzzle rules")
logging.debug("exporting solver assertions to file:")
t = time()
file_name = f"{WIDTH}_{HEIGHT}.smt"
with open(file_name, "w") as f:
    f.write(s.to_smt2())
logging.debug(f"{time() - t:f} seconds")

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
        value = ("" if value else "un") + "shaded"
    print(f"{key} will be {value}")

free_terms = [
    grid[index] for index in np.ndindex(*grid.shape) if index not in CONSTANTS
]
solutions = all_smt(s, free_terms)
# for i, m in enumerate(solutions, start=1):
#     print(i)
# exit(0)
sl = islice(solutions, SOLUTION_COUNT)

t = time()
for i, m in enumerate(sl, start=1):
    logging.debug(f"{time() - t:f} seconds")
    m: ModelRef
    print(f"are_shaded_neighbors: {m.get_interp(are_shaded_neighbors)}")
    # print(f"transitive closure interp: {m.get_interp(shaded_path)}")
    eval_bool_func = np.vectorize(
        lambda expr: is_true(m.eval(expr, model_completion=True))
    )
    eval_int_func = np.vectorize(
        lambda expr: m.eval(expr, model_completion=True).as_long()
    )
    shading = eval_bool_func(grid)
    numbers = eval_int_func(view)
    cell_display_func = np.vectorize(cell_display_l(shading, numbers))

    constant_numbers = np.empty(grid.shape, dtype=str)
    for index in np.ndindex(*constant_numbers.shape):
        constant_numbers[index] = " "
    for key, value in CONSTANTS.items():
        if type(value) is int:
            constant_numbers[key] = str(value)

    shaded_or_constant_number_display = np.vectorize(
        shaded_or_display_l(shading, constant_numbers)
    )
    cells = np.fromfunction(shaded_or_constant_number_display, grid.shape, dtype=int)
    print(f"Solution #{i}:")
    mat_display(cells)
    print()
    t = time()
if not next(solutions, None):
    print("No more solutions")
