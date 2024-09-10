import logging
from z3 import *
from tools import *
from time import time
from itertools import islice, accumulate
import numpy as np

logging.basicConfig(level=logging.DEBUG)

SOLUTION_COUNT = 20  # None for all solutions
WIDTH, HEIGHT = 1, 1
SIZE = WIDTH * HEIGHT
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")
get_coordinate, get_cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

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

Coordinate, mk_coordinate, (coordinate_1, coordinate_2) = TupleSort(
    "coordinate", [IntSort(), IntSort()]
)
cell_coordinates = Function("cell_coordinates", Cell, Coordinate)

for index in range(SIZE):
    x, y = get_coordinate(index)
    s.add(cell_coordinates(CellConsts[index]) == mk_coordinate(x, y))

shading = Function("shading", Cell, BoolSort())

grid = np.empty((WIDTH, HEIGHT), dtype=ExprRef)
for index in range(SIZE):
    x, y = get_coordinate(index)
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

logging.debug("constructing: shaded path")
t = time()

are_neighbors = Function("are_neighbors", Cell, Cell, BoolSort())

c1_q, c2_q = Consts("c1_q, c2_q", Cell)
c1_coordinate_q, c2_coordinate_q = cell_coordinates(c1_q), cell_coordinates(c2_q)
c1_x_q, c1_y_q = coordinate_1(c1_coordinate_q), coordinate_2(c1_coordinate_q)
c2_x_q, c2_y_q = coordinate_1(c2_coordinate_q), coordinate_2(c2_coordinate_q)
same_row_q = c1_y_q == c2_y_q
neighboring_rows_q = Or(c1_y_q + 1 == c2_y_q, c2_y_q + 1 == c1_y_q)
same_column_q = c1_x_q == c2_x_q
neighboring_columns_q = Or(c1_x_q + 1 == c2_x_q, c2_x_q + 1 == c1_x_q)
are_neighbors_q = Or(
    And(same_row_q, neighboring_columns_q), And(same_column_q, neighboring_rows_q)
)
s.add(ForAll([c1_q, c2_q], are_neighbors(c1_q, c2_q) == are_neighbors_q))

are_shaded_neighbors = Function("are_shaded_neighbors", Cell, Cell, BoolSort())

both_shaded_q = And(shading(c1_q), shading(c2_q))
s.add(
    ForAll(
        [c1_q, c2_q],
        are_shaded_neighbors(c1_q, c2_q)
        == And(are_neighbors(c1_q, c2_q), both_shaded_q),
    )
)

shaded_path = TransitiveClosure(are_shaded_neighbors)
logging.debug(f"{time() - t:f} seconds")

logging.debug("constraining: shaded path")
t = time()

# for i, j in np.ndindex((SIZE, SIZE)):
#     c_i, c_j = CellConsts[i], CellConsts[j]
#     s.add(Implies(And(shading(c_i), shading(c_j)), shaded_path(c_i, c_j)))
s.add(ForAll([c1_q, c2_q], both_shaded_q == shaded_path(c1_q, c2_q)))

logging.debug(f"{time() - t:f} seconds")

t1 = time()
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
sl = islice(solutions, SOLUTION_COUNT)

t = time()
for i, m in enumerate(sl, start=1):
    logging.debug(f"{time() - t:f} seconds")
    m: ModelRef
    # print(f"are_shaded_neighbors: {m.get_interp(are_shaded_neighbors)}")
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
