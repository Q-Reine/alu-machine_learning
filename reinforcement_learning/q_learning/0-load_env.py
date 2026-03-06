#!/usr/bin/env python3
"""
This module contains the function load_frozen_lake
for loading the FrozenLake environment from OpenAI's gym.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLakeEnv environment.

    Args:
        desc (list): Optional; a custom description of the map as a list of lists.
        map_name (str): Optional; a pre-made map to load.
        is_slippery (bool): Determines if the ice is slippery.

    Returns:
        gym.Env: The FrozenLake environment.
    """
    # Use FrozenLake-v0 instead of FrozenLake-v1
    return gym.make("FrozenLake-v0", desc=desc, map_name=map_name, is_slippery=is_slippery)

