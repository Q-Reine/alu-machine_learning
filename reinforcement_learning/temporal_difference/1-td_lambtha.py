#!/usr/bin/env python3
"""
TD(λ) algorithm implementation for reinforcement learning.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Args:
        env: The OpenAI environment instance.
        V (numpy.ndarray): Value estimate of shape (s,).
        policy (function): Function that takes in a state and returns the next action to take.
        lambtha (float): The eligibility trace factor.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: Updated value estimate V.
    """
    for episode in range(episodes):
        state = env.reset()
        eligibility_traces = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # Temporal Difference Error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility traces
            eligibility_traces[state] += 1

            # Update value function using eligibility traces
            V += alpha * td_error * eligibility_traces

            # Decay eligibility traces
            eligibility_traces *= gamma * lambtha

            if done:
                break

            state = next_state

    return V
