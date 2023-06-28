from z3 import *
from tools import coordinate_l, index_l
from itertools import starmap
from functools import partial,reduce


compose = lambda *fs: reduce(lambda f,g: lambda *a, **kw: f(g(*a, **kw)), fs)

width, height = Ints("width height")

coordinate = coordinate_l(width)
index = index_l(width)

board = Array("1d-board", IntSort(), BoolSort())

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
# board = Array(f"kth-view", IntSort(), IntSort())


##########################
if __name__ == "__main__":
    WIDTH, HEIGHT = 3, 3

    print(rule_no_2x2.sexpr())
    index = index_l(WIDTH)
    solver = Solver()

    solver.add(width == WIDTH)
    solver.add(height == HEIGHT)
    solver.add(rule_no_2x2)

    solver.add(board[index(10,0)] == True)
    solver.add(board[index(10,1)] == True)
    solver.add(board[index(11,0)] == True)
    solver.add(board[index(11,1)] == True)

    if solver.check() == unsat:
        print("unsat")
        exit(1)

    model = solver.model()
    from tools import bool_display, mat_display
    import numpy as np

    eval_bool_mat = np.vectorize(
        compose(is_true, lambda e: model.eval(e, model_completion=True))
    )
    bool_to_char = np.vectorize(bool_display)

    board_elements = [[board[index(x,y)] for x in range(WIDTH)] for y in range(HEIGHT)]
    shading = eval_bool_mat(board_elements)
    mat_display(bool_to_char(shading))
