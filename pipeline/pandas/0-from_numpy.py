#!/usr/bin/env python3
"""
Creates a pd.DataFrame from a np.ndarray
"""

import numpy as np
import pandas as pd
import string

def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.
    Args:
        array (np.ndarray): The input numpy array.
    Returns:
        pd.DataFrame: The newly created DataFrame with columns labeled A, B, C, etc.
    """
    # Generate column labels
    num_columns = array.shape[1]
    columns = list(string.ascii_uppercase[:num_columns])

    # Create DataFrame
    df = pd.DataFrame(array, columns=columns)

    return df
