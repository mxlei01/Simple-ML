"""Implements LogLikelihood."""

import numpy as np


class LogLikelihood:

    """Class for computing log likelihoods.

    Computes log likelihood, which can be used to calculate log likelihood on logistic regression algorithms.

    """

    @staticmethod
    def log_likelihood(feature_matrix, label, coefficients):
        """Compute log likelihood.

        Used to compute the log likelihood, which is based on:
            ℓℓ(w)=∑^N_i=1((1[yi=+1]−1)wTh(xi)−ln(1+exp(−w^Th(xi))))
        Where,
            1[yi=+1]−1: An indicator function of yi=+1.
            w: Coefficients.
            h(xi): Nth feature.

        Args:
            feature_matrix (numpy.ndarray): Feature matrix.
            label (numpy.array): Labels of the feature matrix.
            coefficients (numpy.array): Coefficients computed using MLE (with or without L1/L2).

        Returns:
            lp (float): Log likelihood.

        """
        # Compute the indicator function 1[yi=+1]
        indicator = (label == +1)

        # Get the score, which is w^t*h(xi)
        scores = np.dot(feature_matrix, coefficients)

        # Compute the log of the score, ln(1+exp(−wTh(xi))
        logexp = np.log(1. + np.exp(-scores))

        # Simple check to prevent overflow
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]

        # Sum over all all the values of indicator*score - logexp
        lp = np.sum((indicator - 1) * scores - logexp)

        return lp

    @staticmethod
    def average_log_likelihood(feature_matrix, label, coefficients):
        """Compute average log likelihood.

        Used to compute the log likelihood, which is based on:
            ℓℓa(w)=(1/N)*∑^N_i=1((1[yi=+1]−1)wTh(xi)−ln(1+exp(−w^Th(xi))))
        Where,
            1[yi=+1]−1: An indicator function of yi=+1.
            w: Coefficients.
            h(xi): Nth feature.
            (1/N): Averages the log likelihood by rows of feature_matrix.

        Args:
            feature_matrix (numpy.ndarray): Feature matrix.
            label (numpy.array): Labels of the feature matrix.
            coefficients (numpy.array): Coefficients computed using MLE (with or without L1/L2).

        Returns:
            lp (float): Average log likelihood.

        """
        # Compute the indicator function 1[yi=+1]
        indicator = (label == +1)

        # Get the score, which is w^t*h(xi)
        scores = np.dot(feature_matrix, coefficients)

        # Compute the log of the score, ln(1+exp(−wTh(xi))
        logexp = np.log(1. + np.exp(-scores))

        # Simple check to prevent overflow
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]

        # Sum over all all the values of indicator*score - logexp
        lp = np.sum((indicator - 1) * scores - logexp) / len(feature_matrix)

        return lp

    @staticmethod
    def log_likelihood_l2_norm(feature_matrix, label, coefficients, l2_penalty):
        """Compute log likelihood with l2 norm.

        Used to compute the log likelihood with l2 norm, which is based on:
            ℓℓ(w)=∑^N_i=1((1[yi=+1]−1)wTh(xi)−ln(1+exp(−w^Th(xi))))-lambda||w||^2_2
        Where,
            1[yi=+1]−1: An indicator function of yi=+1.
            w: Coefficients.
            h(xi): Nth feature.
            lambda: L2_penalty.

        Args:
            feature_matrix (numpy.ndarray): Feature matrix.
            label (numpy.array): Labels of the feature matrix.
            coefficients (numpy.array): Coefficients computed using MLE (with or without L1/L2).
            l2_penalty (float): L2 penalty value.

        Returns:
            lp (float): Log likelihood with l2 norm.

        """
        # Compute the indicator function 1[yi=+1]
        indicator = (label == +1)

        # Get the score, which is w^t*h(xi)
        score = np.dot(feature_matrix, coefficients)

        # Sum over all of the values of indicator*score - logexp and minus the l2 penalty and summing all the
        # coefficient while squared
        lp = np.sum((indicator - 1) * score - np.log(1. + np.exp(-score))) - l2_penalty * np.sum(coefficients[1:] ** 2)

        return lp
