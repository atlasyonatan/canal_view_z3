from sys import path

path.append(r"C:\Program Files\z3-4.13.0-x64-win\bin\python")

import logging
from z3 import *
from time import time
import numpy as np
from itertools import product


WIDTH, HEIGHT = 1, 2
SIZE = WIDTH * HEIGHT

grid_shape = (WIDTH, HEIGHT)

Bool = BoolSort()
X = FiniteDomainSort("x", WIDTH)
Y = FiniteDomainSort("y", HEIGHT)

# X, mk_x = EnumSort("x", [f'{x}' for x in range(WIDTH)])
# Y, mk_y = EnumSort("y", [f'{y}' for y in range(HEIGHT)])

# x_neighbors = Function('x_neighbors', X, X, Bool)
# y_neighbors = Function('y_neighbors', Y, Y, Bool)


# mk_x[i, i+1]

# for w in np.lib.stride_tricks.sliding_window_view(mk_x, (2, )):


# Int = IntSort()


Node, mk_node, (first, second) = TupleSort("node", [X, Y])

shaded = Function("shaded", Node, Bool)

# def in_bounds(pair):
# return coordinate_in_bounds((first(pair), second(pair)), grid_shape)
# Node = DeclareSort("Node")
# a, b, c = Consts('a b c', Node)

# R = Function('R', Node,Node, Bool)
neighbors = Function("neighbors", Node, Node, Bool)

s = Solver()

# for i in range(len(mk_x) - 1):
#     s.add(x_neighbors(mk_x[]) mk_x[i], )

n1, n2 = Consts("n1 n2", Node)

# for (ix, x), (iy, y) in product(enumerate(mk_x), enumerate(mk_y)):
# for (ix2, x2), (iy2, y2) in product(enumerate(mk_x), enumerate(mk_y)):
# if ix == ix2 and abs
# for p2 in map(mk_node, product(mk_x, mk_y)):
# s.add(neighbors(p1, p2) == )


# s.add(
#     ForAll(
#         [n1, n2],
#         neighbors
#         == Or(
#             And(first(n1) == first(n2), abs(second(n1) - second(n2)) == 1),
#             And(second(n1) == second(n2), abs(first(n1) - first(n2)) == 1),
#         ),
#     )
# )

# s.add(
#     ForAll(
#         [n1, n2],
#         neighbors
#         == Or(
#             And(first(n1) == first(n2), abs(second(n1) - second(n2)) == 1),
#             And(second(n1) == second(n2), abs(first(n1) - first(n2)) == 1),
#         ),
#     )
# )

# s.add(R(mk_pair(0,0), mk_pair(0,2)))

# are_neighbors = Or(
#     And(first(p1) == first(p2), abs(second(p1) - second(p2)) == 1),
#     And(second(p1) == second(p2), abs(first(p1) - first(p2)) == 1),
# )
# s.add(ForAll([p1, p2], R(p1, p2) == are_neighbors))


# s.add(R(mk_pair(0, 0), mk_pair(1, 0)) == True)
# s.add(R(mk_pair(1, 0), mk_pair(2, 0)) == True)
# s.add(R)

# s.add(R(a,b))
# s.add(R(b,c))


shaded_connected = TransitiveClosure(neighbors)

# s.add(TC_R(mk_pair(mk_x[0], mk_y[0]), mk_pair(mk_x[0], mk_y[2])))
s.add(ForAll([n1, n2], shaded_connected(n1, n2)))
# get_coordinate, get_cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)
# for x1, y1 in product(range(WIDTH), range(HEIGHT)):
#     for x2, y2 in product(range(WIDTH), range(HEIGHT)):
#         n1 = mk_node(FiniteDomainVal(x1, X), FiniteDomainVal(y1, Y))
#         n2 = mk_node(FiniteDomainVal(x2, X), FiniteDomainVal(y2, Y))
#         s.add(shaded_connected(n1, n2))

# for n1 in map(mk_node, product(mk_x, mk_y)):
#     for n2 in map(mk_node, product(mk_x, mk_y)):
#         s.add(shaded_connected(n1, n2))


# print("done loop")

# s.add(Not(TC_R(mk_pair(0, 0), mk_pair(2,0))))
# s.add(Not(TC_R(a, c)))

print(s.check())
m = s.model()

eval_bool_func = np.vectorize(
    lambda expr: is_false(m.eval(expr, model_completion=True))
)
# print(eval_bool_func(TC_R(mk_pair(0, 0), mk_pair(2, 0))))
# print(m.eval(TC_R(mk_pair(0, 0), mk_pair(2, 0))))
print(m[neighbors])
