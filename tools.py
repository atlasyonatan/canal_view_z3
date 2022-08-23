from z3 import If, And, Or, Not, BoolRef, Int, sat, is_true

# implement python operators for some z3 objects
BoolRef.__radd__ = lambda self, other: self + other
BoolRef.__add__ = lambda self, other: And(self, other)
BoolRef.__rmul__ = lambda self, other: self * other
BoolRef.__mul__ = lambda self, other: Or(self, other)


def coordinate_l(width):
    return lambda cell_number: (cell_number % width, cell_number // width)


def cell_number_l(width):
    return lambda x, y: y + x * width


def dot_product_l(mat1, mat2, sum_op, mul_op):
    n = len(mat1[0])
    if n != len(mat2):
        raise ValueError("dim2 of mat1 must be the same length as dim1 of mat2")
    return lambda i, j: sum_op([mul_op(mat1[i][k], mat2[k][j]) for k in range(n)])


def z3bool_to_int(b):
    return If(b, 1, 0)


#
# def manhatten(x1, y1, x2, y2):
#     return abs(x2 - x1) + abs(y2 - y1)


# def get_neighbours(x, y):
#     n = []
#     if x > 0:
#         n.append((x - 1, y))
#     if x < WIDTH - 1:
#         n.append((x + 1, y))
#     if y > 0:
#         n.append((x, y - 1))
#     if y < HEIGHT - 1:
#         n.append((x, y + 1))
#     return n


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

def mat_display(mat, display_item):
    width = len(mat[0])
    height = len(mat)
    print('x', ' '.join([str(x) for x in range(width)]))
    for x in range(width):
        print(x, end=" ")
        for y in range(height):
            print(display_item(x, y), end=" ")
        print()


def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))

    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))

    def all_smt_rec(terms):
        if sat == s.check():
            m = s.model()
            yield m
            for i in range(len(terms)):
                s.push()
                block_term(s, m, terms[i])
                for j in range(i):
                    fix_term(s, m, terms[j])
                yield from all_smt_rec(terms[i:])
                s.pop()

    yield from all_smt_rec(list(initial_terms))
