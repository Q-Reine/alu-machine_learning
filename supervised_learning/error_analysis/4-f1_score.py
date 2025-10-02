#!/usr/bin/env python3
"""
Calculates the F1 score for each class in a confusion matrix.
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculating the F1 score of a confusion matrix"""
    a = precision(confusion)
    b = sensitivity(confusion)
    return 2 * (a * b) / (a + b)
