#!/usr/bin/env -S python
from z3 import *
from itertools import accumulate
from time import time

HEIGHT, WIDTH = 3, 3
SIZE = WIDTH * HEIGHT

s = Solver()
grid = [[Bool(f'cell_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH)]
adjacency = [[Bool(f'adjacency_{r}_{c}') for c in range(SIZE)] for r in range(SIZE)]
# numbers = [ [Int(f'number_{r}_{c}') for c in range(HEIGHT)] for r in range(WIDTH) ]


# for x in range(0, WIDTH):
#    for y in range(0, HEIGHT):
#        # a cell with a number can't be shaded
#        s.add( Implies(numbers[x][y] > 0, grid[x][y] == False) )

# 2x2 shaded cells are not allowed
for x in range(0, WIDTH - 1):
    for y in range(0, HEIGHT - 1):
        s.add(Not(And([grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]])))


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

def dot_generator(A, B):
    n = len(A[0])
    return lambda i, j: Or([And(A[i][k], B[k][j]) for k in range(n)])


def dot(A, B):
    g = dot_generator(A, B)
    n = len(A[0])
    return [[g(i, j) for i in range(n)] for j in range(n)]


# Aks is an accumulated list of A^k for all k in [1 .. SIZE-1]
# such that Aks[i] = A^(i+1)
Aks = list(accumulate(range(SIZE - 2), lambda acc, _: dot(acc, adjacency), initial=adjacency))

# for i, a in enumerate(Aks):
#    print(i, a)

for i in range(0, SIZE - 1):
    for j in range(i + 1, SIZE):
        # ensure SA is non zero for shaded cell pairs
        # SA records the existance of any paths between two cells

        s.add(And(grid[i % WIDTH][i // WIDTH], grid[j % WIDTH][j // WIDTH]) == Or(
            [Aks[k][i][j] for k in range(0, SIZE - 2)]))

constant = [(0, 0), (2, 2), (0, 4)]
for c in constant:
    print(c, " will be shaded")
    s.add(grid[c[0]][c[1]] == True)

# are we SAT?
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