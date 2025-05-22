#!/usr/bin/env python3
"""
    Flips a 2D matrix over its main diagonal.
"""


def matrix_transpose(matrix):
    """ returns new matrix that is a transpose of the given 2D matrix """
    trans_matrix = [[matrix[j][i] for j in range(len(matrix))]
                    for i in range(len(matrix[0]))]
    return trans_matrix
