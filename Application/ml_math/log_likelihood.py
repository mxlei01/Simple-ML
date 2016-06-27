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

        # Compuete the indicator function 1[yi=+1]
        indicator = (label==+1)

        # Get the score, which is w^t*h(xi)
        scores = np.dot(feature_matrix, coefficients)

        # Compute the log of the score, ln(1+exp(−wTh(xi))
        logexp = np.log(1. + np.exp(-scores))

        # Simple check to prevent overflow
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]

        # Sum over all all the values of indicator*score - logxep
        lp = np.sum((indicator-1)*scores - logexp)

        return lp
