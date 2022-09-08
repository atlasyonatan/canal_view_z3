import itertools

from z3 import And, Or, sat, Solver


def coordinate_l(width):
    return lambda cell_number: (cell_number % width, cell_number // width)


def cell_number_l(width):
    return lambda x, y: x + y * width


def z3_bool_mat_mul(mat1, mat2):
    return lambda i, j: simplify_or([simplify_and(mat1[i][k], mat2[k][j]) for k in range(len(mat1))])


def simplify_and(*terms):
    if any(t is False for t in terms):
        return False
    return And(terms)


def simplify_or(terms):
    a = [t for t in terms if t is not False]
    if len(a) == 0:
        return False
    return Or(a)


def z3_bool_mat_sum(mats):
    return lambda *index: simplify_or([mat[index] for mat in mats])


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


def fix_term_expr(m, t):
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


def puzzles(s, constraints, free_terms, stop=None):
    def puzzles_rec(constrained, start):
        s_ = Solver()
        s_.assert_exprs(s.assertions())
        count = i_len(itertools.islice(all_smt(s_, free_terms), stop))
        if count == 0:
            return
        if type(stop) is not int or count < stop:
            yield constrained
        for i in range(start, len(constraints)):
            s.push()
            constraint = constraints[i]
            s.add(constraint)
            yield from puzzles_rec(constrained + [i], i + 1)
            s.pop()

    return puzzles_rec([], 0)


def i_len(iterator):
    return sum(1 for _ in iterator)
