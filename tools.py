def coordinate_l(width):
    return lambda cell_number: (cell_number % width, cell_number // width)


def cell_number_l(width):
    return lambda x, y: y + x * width


def dot_product_l(mat1, mat2, sum_op, mul_op):
    n = len(mat1[0])
    if n != len(mat2):
        raise ValueError("dim2 of mat1 must be the same length as dim1 of mat2")
    return lambda i, j: sum_op([mul_op(mat1[i][k], mat2[k][j]) for k in range(n)])
