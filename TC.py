import logging
from z3 import *
from tools import *
from time import time
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# #######################
A, AConsts = EnumSort("A", [str(i) for i in range(3)])
B = BoolSort()
R = Function("R", A, A, B)
TC_R = TransitiveClosure(R)
s = Solver()
a, b, c = Consts("a b c", A)
# s.add(R(a, b))
# s.add(Not(R(a, c)))
s.add(ForAll([a, b], R(a, b) == (A == B)))
s.add(ForAll([a, b], TC_R(a, b)))
print(s.check())  # produces unsat
m = s.model()
print(m.get_interp(TC_R))
exit(0)
# #######################

WIDTH, HEIGHT = 3, 3
SIZE = WIDTH * HEIGHT
# get_coordinate, get_cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)

grid_shape = (WIDTH, HEIGHT)

Int = IntSort()
Bool = BoolSort()

Pair, mk_pair, (first, second) = TupleSort("pair", [Int, Int])

grid = Function("grid", Pair, Bool)


def in_bounds(pair):
    return coordinate_in_bounds((first(pair), second(pair)), grid_shape)


shaded_neighbors = Function("shaded_neighbors", Pair, Pair, Bool)

s = Solver()

p1, p2 = Consts("p1 p2", Pair)

are_in_bounds = And(in_bounds(p1), in_bounds(p2))
are_shaded = And(grid(p1), grid(p2))
are_neighbors = Or(
    And(first(p1) == first(p2), z3_abs(second(p1) - second(p2)) == 1),
    And(second(p1) == second(p2), z3_abs(first(p1) - first(p2)) == 1),
)
s.add(
    ForAll(
        [p1, p2],
        shaded_neighbors(p1, p2) == And(are_in_bounds, are_shaded, are_neighbors),
        # shaded_neighbors(p1, p2) == And(are_in_bounds, are_neighbors),
    )
)


shaded_connected = TransitiveClosure(shaded_neighbors)

# s.add(ForAll([p1, p2], shaded_connected(p1, p2) == And(are_in_bounds, are_shaded)))
s.add(
    ForAll([p1, p2], Implies(And(are_in_bounds, are_shaded), shaded_connected(p1, p2)))
)

shaded = [(0, 0), (2, 0), (0, 1)]
unshaded = [(1, 0)]

for c in shaded:
    s.add(grid(mk_pair(*c)))

for c in unshaded:
    s.add(Not(grid(mk_pair(*c))))

# s.add(Not(shaded_connected(mk_pair(0,0), mk_pair(2,0))))

logging.debug("checking sat")
t = time()
sat_result = s.check()
t1 = time()
logging.debug(f"{t1 - t:f} seconds")

if sat_result == unsat:
    print("We are not SAT D:")
    exit(1)

logging.debug("exporting solver assertions to file:")
t = time()
file_name = f"{WIDTH}_{HEIGHT}.smt"
with open(file_name, "w") as f:
    f.write(s.to_smt2())
logging.debug(f"{time() - t:f} seconds")

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
# i,j = (pair_grid[shaded[0]], pair_grid[shaded[1]])
# print(eval_bool_func(shaded_neighbors(i, j)))
# print(eval_bool_func(shaded_connected(i, j)))

print(eval_bool_func(shaded_neighbors(mk_pair(0, 0), mk_pair(0, 1))))
print(eval_bool_func(shaded_neighbors(mk_pair(0, 1), mk_pair(0, 0))))

# print(eval_bool_func(shaded_neighbors(mk_pair(0, 0), mk_pair(1, 0))))
# print(eval_bool_func(shaded_neighbors(mk_pair(1, 0), mk_pair(2, 0))))
# print(eval_bool_func(shaded_neighbors(mk_pair(0, 0), mk_pair(2, 0))))
# print(eval_bool_func(shaded_connected(mk_pair(0, 0), mk_pair(2, 0))))
