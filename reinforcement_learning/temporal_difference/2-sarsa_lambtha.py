#!/usr/bin/env python3
"""
SARSA(λ) algorithm implementation.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon, n_actions):
    """
    Selects an action using the epsilon-greedy strategy.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) algorithm.
    
    Parameters:
    - env: OpenAI environment instance.
    - Q: numpy.ndarray of shape (s, a), containing Q-table.
    - lambtha: Eligibility trace factor.
    - episodes: Total number of episodes to train.
    - max_steps: Maximum steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.
    - epsilon: Initial epsilon for epsilon-greedy policy.
    - min_epsilon: Minimum epsilon value.
    - epsilon_decay: Decay rate for epsilon.
    
    Returns:
    - Updated Q-table (numpy.ndarray).
    """
    n_states, n_actions = Q.shape

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            # Compute TD error
            td_error = (reward +
                        gamma * Q[next_state, next_action] -
                        Q[state, action])

            # Update eligibility trace
            eligibility[state, action] += 1

            # Update Q values
            Q += alpha * td_error * eligibility

            # Decay eligibility trace
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action

            if done:
                break

        # Update epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
