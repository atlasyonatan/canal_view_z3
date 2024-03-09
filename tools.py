from z3 import And, Or, sat, If


# implement python operators for some z3 objects
# z3.BoolRef.__radd__ = lambda self, other: self + other
# z3.BoolRef.__add__ = lambda self, other: And(self, other)
# z3.BoolRef.__rmul__ = lambda self, other: self * other
# z3.BoolRef.__mul__ = lambda self, other: Or(self, other)


def coordinate_l(width):
    return lambda cell_number: (cell_number % width, cell_number // width)


def cell_number_l(width):
    return lambda x, y: x + y * width


def z3_bool_mat_mul(mat1, mat2):
    return lambda i, j: simplify_or(
        [simplify_and(mat1[i][k], mat2[k][j]) for k in range(len(mat1))]
    )


def simplify_and(*terms):
    if any(t is False for t in terms):
        return False
    return And(terms)


def simplify_or(terms):
    a = [t for t in terms if t is not False]
    if len(a) == 0:
        return False
    return Or(a)

def abs(x):
    return If(x >= 0, x, -x)

def coordinate_in_bounds(coordinate, shape):
    x, y = coordinate
    width, height = shape
    return And(x >= 0, x < width, y >= 0, y < height)

def z3_bool_mat_sum(mats):
    return lambda *index: simplify_or([mat[index] for mat in mats])


def mat_display(mat):
    height = len(mat[0])
    width = len(mat)
    print("x", " ".join([str(x) for x in range(width)]))
    for y in range(height):
        print(y, end=" ")
        for x in range(width):
            print(mat[x][y], end=" ")
        print()


def bool_display(v):
    return "#" if v else " "


def cell_display_l(shading, numbers):
    return lambda *index: "#" if shading[index] else str(numbers[index])
    # return lambda *index: "#" if shading[index] else ' '


def shaded_or_display_l(shading, other):
    return lambda *index: "#" if shading[index] else other[index]
    # return lambda *index: "#" if shading[index] else ' '


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


def get_cardinal_neighbors(coordinate, shape):
    x, y = coordinate
    width, height = shape

    # West
    if 0 <= x - 1 < width:
        yield (x - 1, y)

    # East
    if 0 <= x + 1 < width:
        yield (x + 1, y)

    # North
    if 0 <= y + 1 < height:
        yield (x, y + 1)

    # South
    if 0 <= y - 1 < height:
        yield (x, y - 1)
