from z3 import Select, Store


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
