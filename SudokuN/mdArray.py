from z3 import Select, Store, And


def md_eval(m, arr, *index):
    f = m.eval(arr)
    for i in index:
        f = m.eval(f[i])
    return f


def md_select(arr, index):
    for i in index:
        arr = Select(arr, i)
    return arr


def md_store(arr, index, value):
    degree = len(index)
    arr_refs = [arr]
    for i in range(degree - 1):
        arr = Select(arr_refs[i], index[i])
        arr_refs.append(arr)
    for i in range(degree - 1, -1, -1):
        arr = arr_refs.pop()
        value = Store(arr, index[i], value)
    return value


def same_position(p1, p2):
    return And([p1[i] == p2[i] for i in range(len(p1))])


def same_value(arr, p1, p2):
    return md_select(arr, p1) == md_select(arr, p2)
