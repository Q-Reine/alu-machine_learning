#!/usr/bin/env python3
"""
Write the function  that :

Observation is a numpy.ndarray of shape (T,) that contains
the index of the observation
T is the number of observations
N is the number of hidden states
M is the number of possible observations
Transition is the initialized transition probabilities, defaulted to None
Emission is the initialized emission probabilities, defaulted to None
Initial is the initiallized starting probabilities, defaulted to None
If Transition, Emission, or Initial is None, initialize the probabilities as
being a uniform distribution
Returns: the converged Transition, Emission, or None, None on failure
"""
import numpy as np


def _forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
    T = Observation.shape[0]
    N = Transition.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        state = np.matmul(F[:, t - 1], Transition)
        F[:, t] = state * Emission[:, Observation[t]]
    return np.sum(F[:, T - 1]), F


def _backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    T = Observation.shape[0]
    N = Transition.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        b = Emission[:, Observation[t + 1]]
        c = B[:, t + 1]
        B[:, t] = np.sum(Transition * b * c, axis=1)
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
    if not isinstance(Observation, np.ndarray) \
            or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) \
            or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    return _forward(Observation, Emission, Transition, Initial)


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    if not isinstance(Observation, np.ndarray) \
            or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) \
            or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    return _backward(Observation, Emission, Transition, Initial)


def baum_welch(Observations, N, M,
               Transition=None, Emission=None, Initial=None):
    """performs the Baum-Welch algorithm for a hidden markov model"""
    try:
        if not isinstance(Observations, np.ndarray) \
                or len(Observations.shape) != 1:
            return None, None
        tol = 1e-10
        T = Observations.shape[0]
        if Transition is None:
            Transition = np.ones((N, N)) / N
        if Emission is None:
            Emission = np.ones((N, M)) / M
        if Initial is None:
            Initial = np.ones((N, 1)) / N
        a = Transition.copy()
        b = Emission.copy()
        cond = False
        norm_a = 0
        norm_b = 0
        while not cond:
            old_norm_a = norm_a
            old_norm_b = norm_b
            old_a = a.copy()
            old_b = b.copy()
            _, alpha = _forward(Observations, b, a, Initial)
            _, beta = _backward(Observations, b, a, Initial)
            xi = np.zeros((N, N, T - 1))
            for t in range(T - 1):
                denominator = (np.dot(np.dot(alpha[:, t].T, a) *
                                      b[:, Observations[t + 1]].T,
                                      beta[:, t + 1]))
                for i in range(N):
                    numerator = (alpha[i, t] * a[i, :] *
                                 b[:, Observations[t + 1]].T *
                                 beta[:, t + 1].T)
                    xi[i, :, t] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            # Add additional T'th element in gamma
            gamma = np.hstack(
                (gamma,
                 np.sum(xi[:, :, T - 2], axis=1).reshape((-1, 1)))
            )
            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, Observations == l], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)))
            norm_a = np.linalg.norm(np.abs(old_a - a))
            norm_b = np.linalg.norm(np.abs(old_b - b))
            cond = (
                (np.abs(old_norm_a - norm_a) < tol) and
                (np.abs(old_norm_b - norm_b) < tol)
            )
        return a, b
    except Exception:
        return None, None
