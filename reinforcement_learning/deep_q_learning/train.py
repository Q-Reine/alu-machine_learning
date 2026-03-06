#!/usr/bin/env python3
"""
Script to play Atari's Breakout using a trained DQN agent.
"""

import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


def build_model(input_shape, n_actions):
    """
    Builds a neural network model for the DQN agent.

    Parameters:
    - input_shape: Shape of the input for the model.
    - n_actions: Number of possible actions.

    Returns:
    - Keras model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + input_shape))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(n_actions, activation="linear"))
    return model


def play_agent(env_name="Breakout-v0"):
    """
    Loads a trained policy network and plays Breakout.

    Parameters:
    - env_name: Name of the Gym environment.

    Displays:
    - A game played by the trained agent.
    """
    # Initialize environment
    env = gym.make(env_name)
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Build model and agent
    model = build_model(input_shape, n_actions)
    memory = SequentialMemory(limit=100000, window_length=1)
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory,
                   nb_steps_warmup=0, target_model_update=1e-2,
                   policy=policy)
    dqn.compile(None)

    # Load trained weights
    dqn.load_weights("policy.h5")

    # Play one episode
    dqn.test(env, nb_episodes=1, visualize=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    play_agent()
