"""Implements LinearRegression."""

import numpy as np


class LinearRegression:

    """Class to compute Linear Regression.

    Linear Regression computes a line that best fit the continuous data using gradient descent.

    """

    @staticmethod
    def gradient_descent(feature_matrix, output, model_parameters):
        """Gradient descent algorithm for linear regression.

        Gradient descent algorithm: w^(t+1) <= w^(t) + 2nH^t(y-Hw).
        Where,
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            H: Feature matrix.
            w: Weight vector.
            y: Input vector.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            output (numpy.array): The output of a dataset.
            model_parameters (dict): A dictionary of model parameters,
                {
                    initial_weights (numpy.array): Initial weights that are used,
                    step_size (float): Step size,
                    tolerance (float or None): Tolerance (or epsilon).
                }

        Returns:
            weights (numpy.array): The final weights after gradient descent.

        """
        # Set Converged to False
        converged = False

        # Make sure that the weights is a numpy array
        weights = np.array(model_parameters["initial_weights"])

        # Loop until converged
        while not converged:
            # Compute (y-Hw)
            error = output - np.dot(feature_matrix, weights)

            # Compute -2H^t(y-Hw)
            gradient = -2*np.dot(np.transpose(feature_matrix), error)

            # Compute w^(t+1) <= w^(t) - n(-2H^t(y-Hw))
            weights -= model_parameters["step_size"]*gradient

            # If the magnitude of the gradient is less than tolerance, then we have converged
            # The formula for magnitude is sum of squared array, and then square root, but numpy
            # already have a norm function
            # Although we use this nice norm function, but
            # recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)
            if np.linalg.norm(gradient) < model_parameters["tolerance"]:

                # Set converged to true so that we stop our while loop
                converged = True

        return weights

    @staticmethod
    def gradient_ascent(feature_matrix, output, model_parameters):
        """Gradient ascent algorithm for linear regression.

        Gradient ascent algorithm: w^(t+1) <= w^(t) - 2nH^t(y-Hw).
        Where,
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            H: Feature matrix.
            w: Weight vector.
            y: Input vector.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            output (numpy.array): The output of a dataset.
            model_parameters (dict): A dictionary of model parameters,
                {
                    initial_weights (numpy.array): Initial weights that are used,
                    step_size (float): Step size,
                    tolerance (float): Tolerance (or epsilon).
                }


        Returns:
            weights (numpy.array): The final weights after gradient ascent.

        """
        # Set Converged to False
        converged = False

        # Make sure that the weights is a numpy array
        weights = np.array(model_parameters["initial_weights"])

        # Loop until converged
        while not converged:
            # Compute (y-Hw)
            error = output - np.dot(feature_matrix, weights)

            # Compute -2H^t(y-Hw)
            gradient = -2*np.dot(np.transpose(feature_matrix), error)

            # Compute w^(t+1) <= w^(t) + n(-2H^t(y-Hw))
            weights += model_parameters["step_size"]*gradient

            # If the magnitude of the gradient is greater than tolerance, then we have converged
            # The formula for magnitude is sum of squared array, and then square root, but numpy
            # already have a norm function
            if np.linalg.norm(gradient) > model_parameters["tolerance"]:

                # Set converged to true so that we stop our while loop
                converged = True

        return weights
