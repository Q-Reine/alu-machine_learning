#!/usr/bin/env python3
""" Defines function that normalizes a matrix
"""


def normalize(X, m, s):
    """ Normalises the input data using the provided
    mean and standard deviation.

    Args:
    X (numpy.ndarray) An array of features in the shape
    (d, nx) where d is the number of data points and nx
    is the number of input features.

    m (numpy.ndarray) An array of means for each feature
    in the shape (nx,) where nx is the number of features.

    s (numpy.ndarray) An array of standard deviations of
    each feature in the shape (nx,) where nx is the
    number of features.

    Returns:
    numpy.ndarray: X normalised by the mean and standard
    deviation
    """
    return (X - m) / s
