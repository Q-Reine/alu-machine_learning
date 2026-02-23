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


def baum_welch(Observations, N, M,
               Transition=None, Emission=None, Initial=None):
    try:
        Observations = np.asarray(Observations, dtype=int)
        T = Observations.shape[0]

        if Transition is None:
            Transition = np.ones((N, N)) / N
        else:
            Transition = np.asarray(Transition, dtype=float)
            Transition /= np.sum(Transition, axis=1, keepdims=True)

        if Emission is None:
            Emission = np.ones((N, M)) / M
        else:
            Emission = np.asarray(Emission, dtype=float)
            Emission /= np.sum(Emission, axis=1, keepdims=True)

        if Initial is None:
            Initial = np.ones((N, 1)) / N
        else:
            Initial = np.asarray(Initial, dtype=float).reshape(-1, 1)
            Initial /= np.sum(Initial)

        max_iter = 300
        tol = 1e-6
        prev_loglik = -np.inf

        for it in range(max_iter):
            loglik, alpha = forward(Observations, Emission, Transition, Initial)
            _, beta = backward(Observations, Emission, Transition, Initial)

            if not np.isfinite(loglik):
                return None, None

            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)

            xi = np.zeros((T-1, N, N))

            for t in range(T-1):
                denom = np.sum(alpha[t] * np.dot(Transition, Emission[:, Observations[t+1]] * beta[t+1]))
                if denom == 0:
                    denom = 1e-150
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = (
                            alpha[t, i] *
                            Transition[i, j] *
                            Emission[j, Observations[t+1]] *
                            beta[t+1, j]
                        ) / denom

            num_a = np.sum(xi, axis=0)
            denom_a = np.sum(gamma[:-1], axis=0)
            new_Transition = num_a / (denom_a[:, np.newaxis] + 1e-150)

            new_Emission = np.zeros((N, M))
            for k in range(M):
                mask = (Observations == k)
                new_Emission[:, k] = np.sum(gamma[mask], axis=0)

            denom_b = np.sum(gamma, axis=0)
            new_Emission /= (denom_b[:, np.newaxis] + 1e-150)

            new_Initial = gamma[0, :].reshape(-1, 1)

            if abs(loglik - prev_loglik) < tol:
                break

            prev_loglik = loglik

            Transition = new_Transition
            Emission = new_Emission
            Initial = new_Initial

        Transition /= np.sum(Transition, axis=1, keepdims=True)
        Emission   /= np.sum(Emission,   axis=1, keepdims=True)

        return Transition, Emission

    except Exception:
        return None, None
