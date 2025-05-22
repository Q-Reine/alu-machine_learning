#!/usr/bin/env python3
""" Function that adds two matrices """


def add_matrices(mat1, mat2):
    """ returns list of integers representing dimensions of given matrix """

    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) == len(mat2):
            return [add_matrices(a, b) for a, b in zip(mat1, mat2)]
    elif isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2
    return None
