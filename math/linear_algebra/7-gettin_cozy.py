#!/usr/bin/env python3
""" Concatenates two matrices along a specific axis """

def cat_matrices2D(mat1, mat2, axis=0):
    """ returns new matrix that is the concatenation of two 2D matrices """
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
    elif axis == 1:
        if len(mat1) == len(mat2):
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
