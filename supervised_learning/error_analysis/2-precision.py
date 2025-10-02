#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix.
"""


import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (classes, classes)
                                where row indices represent the
                                and column labels.

    Returns:
        np.ndarray: Array of shape (classes,).
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precision_values = true_positives / (true_positives + false_positives)
    return precision_values
