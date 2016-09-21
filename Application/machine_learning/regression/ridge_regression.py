import numpy as np


class RidgeRegression:
    """Class to compute Ridge Regression.

    Ridge Regression is essentially L2 Norm with Linear Regression.

    """

    @staticmethod
    def gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance, l2_penalty, max_iteration=100):
        """Gradient descent algorithm for ridge regression.

        Gradient descent algorithm:  w^(t+1) <= w^(t) - 2nH^t(y-Hw) + l2_penalty*2*w(t).
        Where,
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            H: Feature matrix.
            w: Weight vector.
            y: Input vector.
            l2_penalty: L2 penalty value.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            output (numpy.array): The output of a dataset.
            initial_weights (numpy.array): Initial weights that are used.
            step_size (int): Step size.
            tolerance (int): Tolerance (or epsilon).
            l2_penalty (float): L2 penalty value.
            max_iteration (int): Maximum iteration to compute.

        Returns:
            weights (numpy.array): The final weights after gradient descent.

        """
        # Set Converged to False
        converged = False

        # Make sure that the weights is a numpy array
        weights = np.array(initial_weights)

        # Start at iteration 0
        iteration = 0

        # Loop until converged or until max iteration
        while not converged and iteration != max_iteration:
            # Compute (y-Hw)
            error = output - np.dot(feature_matrix, weights)

            # Compute -2H^t(y-Hw)
            gradient = -2*np.dot(np.transpose(feature_matrix), error)

            # Remember the intercept's gradient
            intercept = gradient[0]

            # Compute gradient(-2H^t(y-Hw))+l2_penalty*2*weights
            gradient += l2_penalty*2*weights

            # We will remove the first gradient's l2_penalty, and only set weights, since the first weight is an
            # intercept
            gradient[0] = intercept

            # Compute w^(t+1) <= w^(t) - n((-2H^t(y-Hw))+l2_penalty*2*weights)
            weights -= step_size*gradient

            # Determine if we have a tolerance value
            if tolerance is not None:
                # If the magnitude of the gradient is less than tolerance, then we have converged
                # The formula for magnitude is sum of squared array, and then square root, but numpy
                # already have a norm function
                # Although we use this nice norm function, but
                # recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)
                if np.linalg.norm(gradient) < tolerance:

                    # Set converged to true so that we stop our while loop
                    converged = True
            iteration += 1

        return weights

    @staticmethod
    def gradient_ascent(feature_matrix, output, initial_weights, step_size, tolerance, l2_penalty, max_iteration=100):
        """Gradient ascent algorithm for ridge regression.

        Gradient ascent algorithm:  w^(t+1) <= w^(t) + 2nH^t(y-Hw) + l2_penalty*2*w(t).
        Where
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            H: Feature matrix.
            w: Weight vector.
            y: Input vector.
            l2_penalty: L2 penalty value.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            output (numpy.array): The output of a dataset.
            initial_weights (numpy.array): Initial weights that are used.
            step_size (int): Step size.
            tolerance (int): Tolerance (or epsilon).
            l2_penalty (float): L2 penalty value.
            max_iteration (int): Maximum iteration to compute.

        Returns:
            weights (numpy.array): The final weights after gradient descent.

        """
        # Set Converged to False
        converged = False

        # Make sure that the weights is a numpy array
        weights = np.array(initial_weights)

        # Start at iteration 0
        iteration = 0

        # Loop until converged or until max iteration
        while not converged and iteration != max_iteration:
            # Compute (y-Hw)
            error = output - np.dot(feature_matrix, weights)

            # Compute -2H^t(y-Hw)
            gradient = -2*np.dot(np.transpose(feature_matrix), error)

            # Remember the intercept's gradient
            intercept = gradient[0]

            # Compute gradient(-2H^t(y-Hw))+l2_penalty*2*weights
            gradient += l2_penalty*2*weights

            # We will remove the first gradient's l2_penalty, and only set weights, since the first weight is an
            # intercept
            gradient[0] = intercept

            # Compute w^(t+1) <= w^(t) + n((-2H^t(y-Hw))+l2_penalty*2*weights)
            weights += step_size*gradient

            # Determine if we have a tolerance value
            if tolerance is not None:
                # If the magnitude of the gradient is less than tolerance, then we have converged
                # The formula for magnitude is sum of squared array, and then square root, but numpy
                # already have a norm function
                # Although we use this nice norm function, but
                # recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)
                if np.linalg.norm(gradient) > tolerance:
                    # Set converged to true so that we stop our while loop
                    converged = True
            iteration += 1

        return weights
