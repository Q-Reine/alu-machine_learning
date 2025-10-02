#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (classes, classes)
                                where row indices represent the correct labels
                                and column indices.

    Returns:
        np.ndarray: Array of shape (classes,).
    """
    true_negatives = np.sum(confusion) - (np.sum(confusion, axis=1)
                                          + np.sum(confusion, axis=0)
                                          - np.diag(confusion))
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)
    specificity_values = true_negatives / (true_negatives + false_positives)
    return specificity_values
