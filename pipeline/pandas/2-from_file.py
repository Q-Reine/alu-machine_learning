#!/usr/bin/env python3
"""
Function to load data from a file into a pd.DataFrame
"""

import pandas as pd

def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.
    Args:
        filename (str): The file to load from.
        delimiter (str): The column separator.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
