from z3 import *
from itertools import starmap
from functools import reduce
from mdArray import sd_to_md, md_to_sd, compose


width, height = Ints("width height")

board = Array("2d-board", IntSort(), BoolSort())
board_shape = (width, height)
board_coordinate = sd_to_md(board_shape)
board_index = md_to_sd(board_shape)

# rule: no 2x2 shaded cells
x, y = Ints("x y")
x_and_y_in_bounds = And(0 <= x, x < width, 0 <= y, y < height)
coordinates_2x2 = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
shaded_2x2 = And(*starmap(compose(board.__getitem__, board_index), (coordinates_2x2)))
rule_no_2x2 = ForAll(
    [x, y], Implies(x_and_y_in_bounds, Not(shaded_2x2))
)  # todo: try without bound


# rule: connectedness
adjacency_pow = Array("3d-adjacency_pow", IntSort(), BoolSort())
board_len = board_shape[0] * board_shape[1]
adjacency_pow_shape = (
    board_len,
    board_len,
    board_len,
)
adjacency_coordinate = sd_to_md(adjacency_pow_shape)
adjacency_index = md_to_sd(adjacency_pow_shape)

i, j, k, q = Ints("i j k q")


abs_z3 = lambda v: If(v >= 0, v, -v)
x1, y1 = board_coordinate(i)
x2, y2 = board_coordinate(j)
are_cardinal_neighbors = Or(
    And(x1 == x2, abs_z3(y1 - y2) == 1), And(y1 == y2, abs_z3(x1 - x2) == 1)
)

i_and_j_in_bounds = And(i >= 0, i < board_len, j >= 0, j < board_len)
k_in_bounds = And(k > 0, k < board_len)
adjacency_pow_1 = ForAll(
    [i, j],
    Implies(
        i_and_j_in_bounds,
        adjacency_pow[adjacency_index(0, i, j)] == are_cardinal_neighbors,
    ),
)

q_in_bounds = And(q >= 0, q < board_len)

exists_path = Exists(
    [q],
    And(
        q_in_bounds,
        adjacency_pow[adjacency_index(0, i, q)],
        adjacency_pow[adjacency_index(k - 1, q, j)],
    ),
)


adjacency_pow_k = ForAll(
    [i, j, k],
    Implies(
        And(k_in_bounds, i_and_j_in_bounds),
        adjacency_pow[adjacency_index(k, i, j)] == exists_path,
    ),
)


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

are_shaded = And(board[i], board[j])
rule_connectedness = ForAll(
    [i, j],
    Implies(
        And(i_and_j_in_bounds, k_in_bounds, are_shaded),
        Exists([k], adjacency_pow[adjacency_index(k, i, j)]),
    ),
)

##########################
if __name__ == "__main__":
    from time import time

    WIDTH, HEIGHT = 2, 2

    solver = Solver()

    solver.add(width == WIDTH)
    solver.add(height == HEIGHT)
    # solver.add(rule_no_2x2)
    solver.add(rule_adjacency_pow)
    solver.add(rule_connectedness)

    solver.add(board[board_index(0, 0)] == True)
    solver.add(board[board_index(0, 1)] == False)
    solver.add(board[board_index(1, 1)] == True)
    solver.add(board[board_index(1, 0)] == False)

    # solver.set(mbqi = True)
    print(solver.sexpr())
    t = time()
    if solver.check() == unsat:
        print("unsat")
        exit(1)
    print(f"sat: {time() - t:f} seconds")
    model = solver.model()

    comp_eval = lambda e: model.eval(e, model_completion=True)
    eval_shape = lambda shape: tuple(map(compose(IntNumRef.as_long, comp_eval), shape))

    from tools import mat_display
    import numpy as np

    board_elements = np.fromfunction(
        np.vectorize(compose(comp_eval, board.__getitem__, board_index)),
        eval_shape(board_shape),
    )

    adjacency = np.fromfunction(
        np.vectorize(compose(comp_eval, adjacency_pow.__getitem__, adjacency_index)),
        eval_shape(adjacency_pow_shape),
    )

    shading = np.vectorize(lambda b: "#" if b else " ")

    # print(shading(board_elements))
    mat_display(shading(board_elements))
    mat_display(shading(adjacency[0]))
    # print(shading(adjacency))
