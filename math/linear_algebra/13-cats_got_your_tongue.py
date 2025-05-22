#!/usr/bin/env python3
""" Concatenation of two matrices along a specific axis using numpy """


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ returns new numpy.ndarray that is concatenation of two matrices """
    return np.concatenate((mat1, mat2), axis=axis)
