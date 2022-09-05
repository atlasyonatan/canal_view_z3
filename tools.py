import z3
from z3 import And, Or, sat


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
    return lambda i, j: Or([And(mat1[i][k], mat2[k][j]) for k in range(len(mat1))])


def z3_bool_mat_sum(mats):
    return lambda *index: Or([mat[index] for mat in mats])


def mat_display(mat):
    height = len(mat[0])
    width = len(mat)
    print('x', ' '.join([str(x) for x in range(width)]))
    for y in range(height):
        print(y, end=" ")
        for x in range(width):
            print(mat[x][y], end=" ")
        print()


def bool_display(v):
    return '#' if v else ' '


def cell_display_l(shading, numbers):
    return lambda *index: "#" if shading[index] else str(numbers[index])
    # return lambda *index: "#" if shading[index] else ' '


def block_term(s, m, t):
    s.add(t != m.eval(t, model_completion=True))


def fix_term(s, m, t):
    s.add(t == m.eval(t, model_completion=True))


def fix_term(m, t):
    return t == m.eval(t, model_completion=True)


def all_smt(s, initial_terms):
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


def yields_above(iterable, n):
    count = 0
    while count <= n and next(iterable, None):
        count += 1
    return count > n

# def accumulate_items

# def redundant(s, terms):
#     if sat != s.check():
#         raise ValueError("The given solver is unsat")
#     m = s.model()
#     s.push()
#
#     # for i in range(len(initial_terms)):
#     #     fix_term(s, m, initial_terms[i])
#
#     def redundant_rec(start):
#         for i in range(start, len(terms)):
#             s.push()
#             block_term(s, m, terms[i])
#             for j in range(i):
#                 fix_term(s, m, terms[j])
#             for j in range(i + 1, len(terms)):
#                 fix_term(s, m, terms[j])
#             check = s.check()
#             s.pop()
#             if sat == check:
#                 yield i
#             else:
#                 s.push()
#                 fix_term(s, m, terms[i])
#                 yield from redundant_rec(i + 1)
#                 s.pop()
#             # if sat == check:
#             #
#             #     had_items = False
#             #     for r in rec:
#             #         had_items = True
#             #         r.append(i)
#             #         yield r
#             #     if not had_items:
#             #         yield [i]
#             #     s.pop()
#
#     yield from redundant_rec(0)
#     s.pop()

# def accumulate()
