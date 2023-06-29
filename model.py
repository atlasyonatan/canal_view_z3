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
adjacency_pow = Array("3d-adjacency_pow", IntSort(), BoolSort())
adjacency_pow_shape = (
    width * height,
    width * height,
    width * height,
)
adjacency_coordinate = sd_to_md(adjacency_pow_shape)
adjacency_index = md_to_sd(adjacency_pow_shape)

i, j, k = Ints("i j k")

abs_z3 = lambda v: If(v >= 0, v, -v)
x1, y1 = coordinate(i)
x2, y2 = coordinate(j)
are_cardinal_neighbors = Or(
    And(x1 == x2, abs_z3(y1 - y2) == 1), And(y1 == y2, abs_z3(x1 - x2) == 1)
)

i_and_j_in_bounds = And(
    i >= 0, i < adjacency_pow_shape[1], j >= 0, j < adjacency_pow_shape[1]
)
k_in_bounds = And(k >= 1, k <= adjacency_pow_shape[0])
are_shaded = And(board[i], board[j])
adjacency_pow_1 = ForAll(
    [i, j],
    Implies(
        i_and_j_in_bounds,
        adjacency_pow[adjacency_index(1, i, j)] == are_cardinal_neighbors,
    ),
)

q = Int("q")
q_in_bounds = And(q >= 0, q < adjacency_pow_shape[1])
def foo(i,j,k):
    return Exists([q], And(adjacency_pow[adjacency_index(1,i,q)], adjacency_pow[adjacency_index(k-1,q,j)]))

adjacency_pow_k = ForAll([i,j,k], Implies(And(i_and_j_in_bounds, k_in_bounds, q_in_bounds), adjacency_pow[adjacency_index(k, i, j)] == foo(i,j,k)))

# q = Int("q")
# vector_multiplication_sum = Array("q_sum", IntSort(), IntSort())
# q_in_bounds = And(i >= 1, i < adjacency_pow_shape[1])

# vector_multiplication = (
#     adjacency_pow[adjacency_index(1, i, q)]
#     * adjacency_pow[adjacency_index(k - 1, q, j)]
# )
# vector_multiplication_sum_0 = vector_multiplication_sum[0] == vector_multiplication
# vector_multiplication_sum_q = ForAll(
#     [q],
#     Implies(
#         q_in_bounds,
#         vector_multiplication_sum[q]
#         == vector_multiplication + vector_multiplication_sum[q - 1],
#     ),
# )

# adjacency_pow_k = ForAll(
#     [k, i, j],
#     Implies(
#         And(i_and_j_in_bounds, k_in_bounds),
#         adjacency_pow[adjacency_index(k, i, j)]
#         == vector_multiplication_sum[adjacency_pow_shape[1] - 1],
#     ),
# )

rule_adjacency_pow = simplify(And(adjacency_pow_1, adjacency_pow_k))

##########################
if __name__ == "__main__":
    WIDTH, HEIGHT = 4, 4

    # board_index = md_to_sd((WIDTH, HEIGHT))
    solver = Solver()

    solver.add(width == WIDTH)
    solver.add(height == HEIGHT)
    # solver.add(rule_no_2x2)
    solver.add(rule_adjacency_pow)

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

    # eval_bool_mat = np.vectorize(compose(is_true, comp_eval))
    # eval_int_mat = np.vectorize(lambda e: comp_eval(e).as_long())

    board_elements = np.fromfunction(np.vectorize(compose(comp_eval, board.__getitem__, ToInt, index)), (WIDTH,HEIGHT))

    # adjacency = [
    #     [
    #         comp_eval(adjacency_pow[adjacency_index(1, x, y)])
    #         for x in range(comp_eval(adjacency_pow_shape[1]).as_long())
    #     ]
    #     for y in range(comp_eval(adjacency_pow_shape[2]).as_long())
    # ]

    adjacency = np.fromfunction(np.vectorize(compose(comp_eval, adjacency_pow.__getitem__, ToInt, adjacency_index)), (WIDTH * WIDTH,WIDTH*WIDTH,HEIGHT*HEIGHT))

    # shading = eval_bool_mat(board_elements)
    # adjacency = eval_int_mat(adjacency_matrix)

    # bool_to_char = np.vectorize(bool_display)

    shading = np.vectorize(lambda b: '#' if b else ' ')
    
    print(shading(board_elements))
    mat_display(shading(adjacency[3]))
    # print(shading(adjacency))
