#!/usr/bin/env python3
"""
Creates a confusion matrix
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (np.ndarray): One-hot numpy array of shape (m, classes)
                             with the correct labels.
        logits (np.ndarray): One-hot numpy array of shape (m, classes)
                             with the predicted labels.

    Returns:
        np.ndarray: Confusion matrix of shape (classes, classes).
    """
    # Convert one-hot encoded labels and logits to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Determine the number of classes
    num_classes = labels.shape[1]

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Populate the confusion matrix
    for true, pred in zip(true_labels, predicted_labels):
        confusion_matrix[true, pred] += 1

    return confusion_matrix
