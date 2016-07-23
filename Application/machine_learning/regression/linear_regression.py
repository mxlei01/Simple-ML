import numpy as np

class LinearRegression:
    # Usage:
    #   Linear Regression is based on: w^(t+1) <= w^(t) + 2nH^t(y-Hw) for gradient descent,
    #   and w^(t+1) <= w^(t) - 2nH^t(y-Hw) for hill climbing
    #       w(t)   : weight at iteration t
    #       w(t+1) : weight at iteration t+1
    #       n      : step size
    #       H      : feature matrix
    #       w      : weight vector
    #       y      : input vector

    def gradient_descent(self, feature_matrix, output, initial_weights, step_size, tolerance):
        # Usage:
        #       Gradient descent algorithm: w^(t+1) <= w^(t) + 2nH^t(y-Hw)
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       output          (numpy array)  : the output of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
        #       step_size       (int)          : step size
        #       tolerance       (int)          : tolerance (or epsilon)
        # Return:
        #       weights         (numpy array)  : the final weights after gradient descent

        # Set Converged to False
        converged = False

        # Make sure that the weights is a numpy array
        weights = np.array(initial_weights)

        # Loop until converged
        while not converged:
            # Compute (y-Hw)
            error = output - np.dot(feature_matrix, weights)

            # Compute -2H^t(y-Hw)
            gradient = -2*np.dot(np.transpose(feature_matrix), error)

            # Compute w^(t+1) <= w^(t) - n(-2H^t(y-Hw))
            weights -= step_size*gradient

            # If the magnitude of the gradient is less than tolerance, then we have converged
            # The formula for magnitude is sum of squared array, and then square root, but numpy
            # already have a norm function
            # Although we use this nice norm function, but
            # recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)
            if np.linalg.norm(gradient) < tolerance:

                # Set converged to true so that we stop our while loop
                converged = True

        # Return the weights
        return weights

    def hill_climbing(self, feature_matrix, output, initial_weights, step_size, tolerance):
        # Usage:
        #       Gradient descent algorithm: w^(t+1) <= w^(t) + 2nH^t(y-Hw)
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       output          (numpy array)  : the output of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
        #       step_size       (int)          : step size
        #       tolerance       (int)          : tolerance (or epsilon)
        # Return:
        #       weights         (numpy array)  : the final weights after gradient descent

        # Set Converged to False
        converged = False

        # Make sure that the weights is a numpy array
        weights = np.array(initial_weights)

        # Loop until converged
        while not converged:
            # Compute (y-Hw)
            error = output - np.dot(feature_matrix, weights)

            # Compute -2H^t(y-Hw)
            gradient = -2*np.dot(np.transpose(feature_matrix), error)

            # Compute w^(t+1) <= w^(t) + n(-2H^t(y-Hw))
            weights += step_size*gradient

            # If the magnitude of the gradient is greater than tolerance, then we have converged
            # The formula for magnitude is sum of squared array, and then square root, but numpy
            # already have a norm function
            if np.linalg.norm(gradient) > tolerance:

                # Set converged to true so that we stop our while loop
                converged = True

        # Return the weights
        return weights