import numpy as np


class LogLikelihood:
    # Usage:
    #   A class to compute log likelihood

    def log_likelihood(self, feature_matrix, label, coefficients):
        # Usage:
        #       Used to compute the log likelihood, which is based on:
        #           ℓℓ(w)=∑^N_i=1((1[yi=+1]−1)wTh(xi)−ln(1+exp(−wTh(xi))))
        #       Where:
        #           1[yi=+1]−1 : is an indicator function of yi=+1
        #           w          : coefficients
        #           h(xi)      : Nth feature
        # Arguments:
        #       feature_matrix (numpy matrix) : feature matrix
        #       label          (numpy array)  : labels of the feature matrix
        #       coefficients   (numpy array)  : coefficients computed using MLE (with or without L1/L2)
        # Returns:
        #       lp (float) : log likelihood

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
        lp = np.sum((indicator-1)*scores - logexp)

        return lp

    def log_likelihood_l2_norm(self, feature_matrix, label, coefficients, l2_penalty):
        # Usage:
        #       Used to compute the log likelihood with l2 norm, which is based on:
        #           ℓℓ(w)=∑^N_i=1((1[yi=+1]−1)wTh(xi)−ln(1+exp(−wTh(xi))))-lambda||w||^2_2
        #       Where:
        #           1[yi=+1]−1 : is an indicator function of yi=+1
        #           w          : coefficients
        #           h(xi)      : Nth feature
        #           lambda     : l2_penalty
        # Arguments:
        #       feature_matrix (numpy matrix) : feature matrix
        #       label          (numpy array)  : labels of the feature matrix
        #       coefficients   (numpy array)  : coefficients computed using MLE (with or without L1/L2)
        #       l2_penalty     (float)        : l2 penalty value
        # Returns:
        #       lp (float) : log likelihood

        # Compute the indicator function 1[yi=+1]
        indicator = (label == +1)

        # Get the score, which is w^t*h(xi)
        scores = np.dot(feature_matrix, coefficients)

        # Sum over all of the values of indicator*score - logexp and minus the l2 penalty and summing all the
        # coefficient while squared
        lp = np.sum((indicator-1)*scores - np.log(1.+np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)

        return lp