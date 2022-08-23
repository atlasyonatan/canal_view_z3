#!/usr/bin/env -S python
import operator
import logging
from z3 import *
from tools import *
from time import time

logging.basicConfig(level=logging.DEBUG)

HEIGHT, WIDTH = 5, 5
SIZE = WIDTH * HEIGHT
coordinate, cell_number = coordinate_l(WIDTH), cell_number_l(WIDTH)
print(f"WIDTH = {WIDTH}, HEIGHT = {HEIGHT}")

s = Solver()

logging.debug("defining value spaces")
t0 = time()
grid = [[Bool(f'cell_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH)]
adjacency = [[[Int(f'adjacency_{k}_{i}_{j}') for j in range(SIZE)] for i in range(SIZE)] for k in range(0, SIZE - 2)]
# numbers = [ [Int(f'number_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH) ]
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

logging.debug("constraining: adjacency int range")
t0 = time()
# limit adjacency z3.Int range
for Ak in adjacency:
    for Aki in Ak:
        for Akij in Aki:
            s.add(And(Akij >= 0, Akij <= SIZE))
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

# for x in range(0, WIDTH):
#    for y in range(0, HEIGHT):
#        # a cell with a number can't be shaded
#        s.add( Implies(numbers[x][y] > 0, grid[x][y] == False) )

logging.debug("constraining: no 2x2 shaded")
t0 = time()
# 2x2 shaded cells are not allowed
for x in range(0, WIDTH - 1):
    for y in range(0, HEIGHT - 1):
        s.add(Not(And([grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]])))
t1 = time()
logging.debug(f"{t1-t0:f} seconds")


def manhatten(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)


def get_neighbours(x, y):
    n = []
    if x > 0:
        n.append((x - 1, y))
    if x < WIDTH - 1:
        n.append((x + 1, y))
    if y > 0:
        n.append((x, y - 1))
    if y < HEIGHT - 1:
        n.append((x, y + 1))
    return n


# for x1 in range(0, WIDTH):
#    for y1 in range(0, HEIGHT):
#         for x2 in range(0, WIDTH):
#            for y2 in range(0, HEIGHT):
#                # self is minimum distance, make no constraints
#                if x1 == x2 and y1 == y2:
#                    continue
#
#                dist = manhatten(x1, y1, x2, y2)
#                conditions = [And(grid[x][y], dist > manhatten(x1, y1, x, y)) for (x, y) in get_neighbours(x2, y2)]
#                s.add(Implies(And([grid[x1][y1], grid[x2][y2]]), Or(conditions)))


# def dot(A, B):
#     g = dot_generator(A, B)
#     n = len(A[0])
#     return [[g(i, j) for i in range(n)] for j in range(n)]


# constrain adjacency 3d matrix where adjacency[k] is equal to

logging.debug("constraining: adjacency matrix of grid")
t0 = time()
# regular Adjacency matrix defined to be 1 when cells i and j in grid are shaded and connected, otherwise 0
k = 0
for x in range(WIDTH):
    for y in range(HEIGHT - 1):
        x1, y1 = x, y
        x2, y2 = x, y + 1
        i = cell_number(x1, y1)
        j = cell_number(x2, y2)
        s.add(adjacency[k][i][j] == If(And(grid[x][y], grid[x][y + 1]), 1, 0))
for x in range(WIDTH - 1):
    for y in range(HEIGHT):
        i = cell_number(x, y)
        j = cell_number(x + 1, y)
        s.add(adjacency[k][i][j] == If(And(grid[x][y], grid[x + 1][y]), 1, 0))
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

logging.debug("constraining: adjacency^k")
t0 = time()
# powers of the adjacency matrix
for k in range(1, len(adjacency)):
    dot = dot_product_l(adjacency[0], adjacency[k - 1], z3.Sum, operator.mul)
    for i in range(SIZE):
        for j in range(SIZE):
            s.add(adjacency[k][i][j] == dot(i, j))
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

logging.debug("constraining: sum of adjacency^k is nonzero for shaded cell pairs")
t0 = time()
# constrain the sum of adjacency^k for k in [1..SIZE-1], is positive for all shaded cells
for i in range(0, SIZE - 1):
    for j in range(i + 1, SIZE):
        x1, y1 = coordinate(i)
        x2, y2 = coordinate(j)
        shaded = And(grid[x1][y1], grid[x2][y2])
        sum_of_adj_k = Sum([adjacency[k][i][j] for k in range(len(adjacency))])
        non_zero = sum_of_adj_k != 0
        s.add(Implies(shaded, non_zero))
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

logging.debug("constraining: constants")
t0 = time()
constant = [(0, 0), (2, 2), (0, 4)]
for c in constant:
    print(c, " will be shaded")
    s.add(grid[c[0]][c[1]])
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

logging.debug("checking sat")
t0 = time()
# are we SAT?
sat_result = s.check()
t1 = time()
logging.debug(f"{t1-t0:f} seconds")

if s.check() == unsat:
    print("We are not SAT D:")
    exit(1)

m = s.model()
print('x', ' '.join([str(x) for x in range(WIDTH)]))
for x in range(WIDTH):
    print(x, end=" ")
    for y in range(HEIGHT):
        print("#" if m.eval(grid[x][y]) else " ", end=" ")
    print()
# print(m)
