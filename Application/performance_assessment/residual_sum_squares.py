"""Implements ResidualSumSquares."""

import numpy as np


class ResidualSumSquares:

    """Computes Residual Sum of Squares.

    Class to compute residual sum of squares, which can be used for linear regression problems.

    """

    @staticmethod
    def residual_sum_squares_regression(output, predicted_output):
        """Computes residual sum of squares.

        Computes for residual sum of squares for regression.

        RSS = (y-Hw)^t * (y-Hw)
        Where,
            y: real output
            Hw: features * coefficients

        Args:
            output (numpy.array): Real output.
            predicted_output (numpy.array): Predicted output.

        Returns:
            float: Residual sum of squares.

        """
        return np.dot(output - predicted_output, output - predicted_output)
