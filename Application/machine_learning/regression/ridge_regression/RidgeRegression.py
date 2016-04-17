import numpy as np

class RidgeRegression:
    # Usage:
    #   Ridge Regression is based on: w^(t+1) <= w^(t) - 2nH^t(y-Hw) + l2_penalty*2*w(t) for gradient descent,
    #   and w^(t+1) <= w^(t) + 2nH^t(y-Hw) + l2_penalty*2*w(t) for hill climbing
    #       w(t)       : weight at iteration t
    #       w(t+1)     : weight at iteration t+1
    #       n          : step size
    #       H          : feature matrix
    #       w          : weight vector
    #       y          : input vector
    #       l2_penalty : l2_penalty value

    def gradient_descent(self, feature_matrix, output, initial_weights, step_size, tolerance, l2_penalty, max_iteration=100):
        # Usage:
        #       Gradient descent algorithm with ridge: w^(t+1) <= w^(t) - 2nH^t(y-Hw) + l2_penalty*2*w(t)
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       output          (numpy array)  : the output of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
        #       step_size       (int)          : step size
        #       tolerance       (int)          : tolerance (or epsilon)
        #       l2_penalty      (double)       : l2_penalty value
        #       max_iteration   (int)          : maximum iteration to compute
        # Return:
        #       weights         (numpy array)  : the final weights after gradient descent

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

            # Compute gradient(-2H^t(y-Hw))+l2_penalty*2*weights
            gradient += l2_penalty*2*weights

            # Compute w^(t+1) <= w^(t) - n((-2H^t(y-Hw))+l2_penalty*2*weights)
            weights -= step_size*gradient
            #print(weights)

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

            # Increment iteration
            iteration += 1

        # Return the weights
        return weights

    def hill_climbing(self, feature_matrix, output, initial_weights, step_size, tolerance, l2_penalty, max_iteration=100):
        # Usage:
        #       Gradient descent algorithm with ridge: w^(t+1) <= w^(t) + 2nH^t(y-Hw) + l2_penalty*2*w(t)
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       output          (numpy array)  : the output of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
        #       step_size       (int)          : step size
        #       tolerance       (int)          : tolerance (or epsilon)
        #       l2_penalty      (double)       : l2_penalty value
        #       max_iteration   (int)          : maximum iteration to compute
        # Return:
        #       weights         (numpy array)  : the final weights after gradient descent

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

            # Compute gradient(-2H^t(y-Hw))+l2_penalty*2*weights
            gradient += l2_penalty*2*weights

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

            # Increment iteration
            iteration += 1

        # Return the weights
        return weights
