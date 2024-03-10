import logging
from z3 import *
from tools import *
from time import time
from itertools import islice, accumulate
import numpy as np

logging.basicConfig(level=logging.DEBUG)


WIDTH, HEIGHT = 3, 3
SIZE = WIDTH * HEIGHT
get_coordinate, get_cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

grid_shape = (WIDTH, HEIGHT)

# grid = np.empty((WIDTH, HEIGHT), dtype=ExprRef)
# for index in np.ndindex(*grid.shape):
#     grid[index] = Bool(f"cell_{'_'.join(str(v) for v in index)}")


Int = IntSort()
Bool = BoolSort()

Pair, mk_pair, (first, second) = TupleSort("pair", [Int, Int])

grid = Function("grid", Pair, Bool)


def in_bounds(pair):
    return coordinate_in_bounds((first(pair), second(pair)), grid_shape)


shaded_neighbors = Function("shaded_neighbors", Pair, Pair, Bool)

s = Solver()

p1, p2 = Consts("p1 p2", Pair)
# s.add(ForAll([p1, p2], shaded_neighbors(p1, p2) == shaded_neighbors(p2, p1)))

print(s.check())

are_in_bounds = And(in_bounds(p1), in_bounds(p2))
are_shaded = And(grid(p1), grid(p2))
are_neighbors = Or(
    And(first(p1) == first(p2), abs(second(p1) - second(p2)) == 1),
    And(second(p1) == second(p2), abs(first(p1) - first(p2)) == 1),
)
s.add(
    ForAll(
        [p1, p2],
        shaded_neighbors(p1, p2) == And(are_in_bounds, are_shaded, are_neighbors),
    )
)


print(s.check())
# for x in range(WIDTH):
#     for y in range(HEIGHT):
#         i_coordinates = (x, y)
#         i_shading = grid[i_coordinates]
#         i = get_cell_number(*i_coordinates)

#         # right neighbor
#         if x + 1 < WIDTH:
#             j_coordinates = (x + 1, y)
#             j_shading = grid[j_coordinates]
#             j = get_cell_number(*j_coordinates)
#             s.add(shaded_neighbors(i, j) == And(i_shading, j_shading))

#         # down neighbor
#         if y + 1 < HEIGHT:
#             j_coordinates = (x, y + 1)
#             i_shading = grid[j_coordinates]
#             j = get_cell_number(*j_coordinates)
#             s.add(shaded_neighbors(i, j) == And(i_shading, j_shading))


# for cell_coordinate, cell_shading in np.ndenumerate(grid):
#     cell_number = get_cell_number(*cell_coordinate)
#     for neighbor_coordinate in get_cardinal_neighbors(cell_coordinate, grid_shape):
#         neighbor_number = get_cell_number(*neighbor_coordinate)
#         neighbor_shading = grid[neighbor_coordinate]

#         s.add(
#             shaded_neighbors(cell_number, neighbor_number)
#             == And(cell_shading, neighbor_shading)
#         )

shaded_connected = TransitiveClosure(shaded_neighbors)

s.add(ForAll([p1, p2], shaded_connected(p1, p2) == And(are_in_bounds, are_shaded)))
# for i in range(SIZE):
#     i_coordinates = get_coordinate(i)
#     i_pair = mk_pair(*i_coordinates)
#     # i_cell_shading = grid[get_coordinate(i)]
#     for j in range(SIZE):
#         i_coordinates = get_coordinate(i)
#         i_pair = mk_pair(*i_coordinates)
#         j_cell_shading = grid[get_coordinate(j)]
#         s.add(Implies(And(i_cell_shading, j_cell_shading), shaded_connected(i, j)))

print(s.check())

shaded = [(0, 0), (2, 0)]
unshaded = [(1, 0), (1, 1)]

for c in shaded:
    s.add(grid(mk_pair(*c)))

for c in unshaded:
    s.add(Not(grid(mk_pair(*c))))

logging.debug("checking sat")
t = time()
sat_result = s.check()
t1 = time()
logging.debug(f"{t1 - t:f} seconds")

if sat_result == unsat:
    print("We are not SAT D:")
    exit(1)

pair_grid = np.zeros(grid_shape, dtype=ExprRef)
for index in np.ndindex(*pair_grid.shape):
    pair_grid[index] = mk_pair(*index)

shading_grid = np.vectorize(lambda expr: grid(expr))(pair_grid)
# print(grid_array)

m = s.model()

eval_bool_func = np.vectorize(lambda expr: is_true(m.eval(expr, model_completion=True)))
# for index in np.ndindex(*eval_grid.shape):
#     x, y = index
#     eval_grid[index] = )
grid_array_eval = eval_bool_func(shading_grid)
# print(grid_array_eval)
shading = np.vectorize(bool_display)(grid_array_eval)
# print(shading)
# grid_display = np.vectorize(lambda *index: bool_display(shading[index]))
# cells = np.fromfunction(grid_display, grid_array.shape, dtype=int)
mat_display(shading)

print(eval_bool_func(shaded_neighbors(pair_grid[shaded[0]], pair_grid[shaded[1]])))
print(eval_bool_func(shaded_connected(pair_grid[shaded[0]], pair_grid[shaded[1]])))
