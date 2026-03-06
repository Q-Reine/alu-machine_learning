#!/usr/bin/env python3
"""
Monte Carlo algorithm implementation for reinforcement learning.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: The OpenAI environment instance.
        V (numpy.ndarray): Value estimate of shape (s,).
        policy (function): Function that takes in a state and returns the next action to take.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: Updated value estimate V.
    """
    for episode in range(episodes):
        state = env.reset()
        episode_history = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_history.append((state, reward))
            state = next_state
            if done:
                break

        # Calculate returns and update V
        G = 0
        visited_states = set()
        for state, reward in reversed(episode_history):
            G = reward + gamma * G
            if state not in visited_states:
                visited_states.add(state)
                # Update rule for the value function
                V[state] += alpha * (G - V[state])

    return V
