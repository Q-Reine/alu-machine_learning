#!/usr/bin/env python3
'''  A function that slices a matrix along specific axes using numpy '''


def np_slice(matrix, axes):
    ''' returns def np_slice(matrix, axes):  that slices matrix along axes   '''
    slices_matrix = [slice(None)] * len(matrix.shape)

    for axis, value in axes.items():
        slices_matrix[axis] = slice(*value)

    return matrix[tuple(slices_matrix)]
