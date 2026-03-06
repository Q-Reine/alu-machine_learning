#!/usr/bin/env python3
""" Q-Learning """

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def epsilon_greedy(Q, state, epsilon):
    """Selects action using epsilon-greedy policy."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])  # Random action (explore)
    return np.argmax(Q[state])  # Greedy action (exploit)


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning."""
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset() 
        total_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)  # gymnasium returns 5 values
            done = terminated or truncated

            # Update reward if the agent falls into a hole
            if done and reward == 0:
                reward = -1

            # Update Q-table using the Q-learning formula
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            total_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(total_reward)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
