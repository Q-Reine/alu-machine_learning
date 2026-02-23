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


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model"""
    if not isinstance(Observations, np.ndarray) \
            or len(Observations.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) \
            or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observations.shape[0]
    N, M = Emission.shape

    a = Transition.copy()
    b = Emission.copy()
    tol = 1e-10
    norm_a = 0
    norm_b = 0
    cond = False

    while not cond:
        old_norm_a = norm_a
        old_norm_b = norm_b
        old_a = a.copy()
        old_b = b.copy()

        # Forward pass
        alpha = np.zeros((N, T))
        alpha[:, 0] = Initial[:, 0] * b[:, Observations[0]]
        for t in range(1, T):
            alpha[:, t] = (np.matmul(alpha[:, t - 1], a) *
                           b[:, Observations[t]])

        # Backward pass
        beta = np.zeros((N, T))
        beta[:, T - 1] = np.ones(N)
        for t in range(T - 2, -1, -1):
            beta[:, t] = np.sum(
                a * b[:, Observations[t + 1]] * beta[:, t + 1],
                axis=1
            )

        # Xi computation
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

        # Gamma and parameter updates
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add T'th element in gamma (sum xi over j at T-2, axis=1)
        gamma = np.hstack(
            (gamma,
             np.sum(xi[:, :, T - 2], axis=1).reshape((-1, 1)))
        )
        denominator = np.sum(gamma, axis=1)
        for l in range(M):
            b[:, l] = np.sum(gamma[:, Observations == l], axis=1)
        b = np.divide(b, denominator.reshape((-1, 1)))

        norm_a = np.linalg.norm(np.abs(old_a - a))
        norm_b = np.linalg.norm(np.abs(old_b - b))
        cond = (
            (np.abs(old_norm_a - norm_a) < tol) and
            (np.abs(old_norm_b - norm_b) < tol)
        )
    return a, b
