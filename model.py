from z3 import *
from tools import coordinate_l, index_l
from itertools import starmap
from functools import reduce
from mdArray import sd_to_md, md_to_sd


compose = lambda *fs: reduce(lambda f, g: lambda *a, **kw: f(g(*a, **kw)), fs)

width, height = Ints("width height")

board = Array("2d-board", IntSort(), BoolSort())
board_shape = (width, height)
coordinate = sd_to_md(board_shape)
index = md_to_sd(board_shape)

# rule: no 2x2 shaded cells
x, y = Ints("x y")
x_and_y_in_bounds = And(0 <= x, x < width, 0 <= y, y < height)
coordinates_2x2 = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
indexes_2x2 = starmap(index, coordinates_2x2)
elements_2x2 = map(board.__getitem__, indexes_2x2)
shaded_2x2 = And(*elements_2x2)
rule_no_2x2 = ForAll(
    [x, y], Implies(x_and_y_in_bounds, Not(shaded_2x2))
)  # todo: try without bound


# rule: connectedness
adjacency_pow_k = Array("3d-adjacency_pow_k", IntSort(), IntSort())
adjacency_pow_k_shape = (
    width * width * height * height,
    width * height,
    width * height,
)
adjacency_coordinate = sd_to_md(adjacency_pow_k_shape)
adjacency_index = md_to_sd(adjacency_pow_k_shape)

i, j = Ints("i j")

abs_z3 = lambda v: If(v >= 0, v, -v)
difference = abs_z3(i - j)
are_cardinal_neighbors = Or(
    difference == width,
    And(difference == 1, abs_z3(i % width - j % height) == 1),
)
i_and_j_in_bounds = And(i>=0, i<width*height, j>=0, j<width*height)
# are_shaded = And(board[i], board[j])
adjacency_pow_1 = ForAll(
    [i, j],
    Implies(i_and_j_in_bounds, adjacency_pow_k[adjacency_index(1, i, j)] == If(are_cardinal_neighbors, 1, 0))
)

##########################
if __name__ == "__main__":
    WIDTH, HEIGHT = 3, 3

    # board_index = md_to_sd((WIDTH, HEIGHT))
    solver = Solver()

    solver.add(width == WIDTH)
    solver.add(height == HEIGHT)
    # solver.add(rule_no_2x2)

    solver.add(adjacency_pow_1)

    print(solver.sexpr())
    # solver.add(board[index(0, 0)] == True)
    # solver.add(board[index(0, 1)] == True)
    # solver.add(board[index(1, 0)] == True)
    # solver.add(board[index(1,1)] == True)

    if solver.check() == unsat:
        print("unsat")
        exit(1)

    model = solver.model()

    comp_eval = lambda e: model.eval(e, model_completion=True)

    board_shape = tuple(map(comp_eval, board_shape))

    # adjacency_shape = tuple(map(comp_eval, adjacency_pow_k_shape[1:]))

    from tools import bool_display, mat_display
    import numpy as np

    eval_bool_mat = np.vectorize(compose(is_true, comp_eval))

    eval_int_mat = np.vectorize(lambda e: comp_eval(e).as_long())

    board_elements = [[board[index(x, y)] for x in range(WIDTH)] for y in range(HEIGHT)]

    adjacency_matrix = [
        [
            adjacency_pow_k[adjacency_index(1, x, y)]
            for x in range(comp_eval(adjacency_pow_k_shape[1]).as_long())
        ]
        for y in range(comp_eval(adjacency_pow_k_shape[2]).as_long())
    ]

    shading = eval_bool_mat(board_elements)
    adjacency = eval_int_mat(adjacency_matrix)

    # bool_to_char = np.vectorize(bool_display)
    # mat_display(bool_to_char(shading))
    mat_display(adjacency)
