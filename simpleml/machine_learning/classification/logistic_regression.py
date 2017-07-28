"""Implements LogisticRegression."""

import numpy as np
import math


class LogisticRegression:

    """Class that has various functions to compute Logistic Regression.

    Logistic Regression is essentially using linear regression techniques to compute classification problems by fitting
    a line between two classes.

    """

    @staticmethod
    def gradient_ascent(feature_matrix, label, model_parameters):
        """Gradient ascent algorithm for Logistic Regression.

        The gradient ascent algorithm: w^(t+1) <= w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w)).
        Where:
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            h_j(X_i): Feature for each row of a specific column.
            1[y=+1]: Indicator function for y=+1.
            P(y=1|x_i,w): Probability of y=1 for x_i using the current weights.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            label (numpy.array): The label of a dataset.
            model_parameters (dict): A dictionary of model parameters,
                {
                    initial_coefficients (numpy.array): Initial weights for the model,
                    step_size (float): Step size,
                    max_iter (int): Amount of Iterations.
                }

        Returns:
            coefficients (numpy.array): The final weights after gradient ascent finishes.

        """
        # Make sure we are using numpy array
        coefficients = np.array(model_parameters["initial_coefficients"])

        # Compute the weights up to max_iter
        for _ in range(model_parameters["max_iter"]):
            # we would need to compute the prediction, which is based on the link function
            #           1
            # -------------------   = P(y=1|x_i,w)
            # 1 + exp(-w^t*h(x_i))
            dot_product_results = np.apply_along_axis(lambda feature, coef: np.dot(np.transpose(coef), feature),
                                                      1, feature_matrix, coefficients)

            # Compute P(y_i = +1 | x_i, w) using the link function
            predictions = [1 / (1 + math.exp(-weight_dot_feature)) for weight_dot_feature in dot_product_results]

            # Compute indicator value for (y_i = +1)
            indicator = (label == +1)

            # Compute the errors as indicator - predictions, (1[y=+1]-P(y=1|x_i,w)
            errors = indicator - predictions

            # Compute the coefficients by using w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            # in matrix form
            # We do a transpose of feature matrix to convert rows into the column data, since the
            # the sigma function works on all the values for a specific column, and we will multiply each
            # row will error, which gives us Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            coefficients = coefficients + model_parameters["step_size"] * np.dot(np.transpose(feature_matrix), errors)

        return coefficients

    @staticmethod
    def stochastic_gradient_ascent(feature_matrix, label, model_parameters):
        """Stochastic gradient ascent algorithm for Logistic Regression.

        The gradient ascent algorithm: w^(t+1) <= w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w)).
        Where:
            Σ^N_b: summation up to batch size b, 1 for SGA, and anything above 1 is BGA (Batch)
            w(t): weight at iteration t.
            w(t+1): weight at iteration t+1.
            n: step size.
            h_j(X_i): feature for each row of a specific column.
            1[y=+1]: indicator function for y=+1.
            P(y=1|x_i,w): probability of y=1 for x_i using the current weights.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            label (numpy.array): The label of a dataset.
            model_parameters (dict): A dictionary of model parameters,
                {
                    initial_coefficients (numpy.array): Initial weights for the model,
                    step_size (float): Step size,
                    batch_size (int): Number of items to sum per iteration,
                    max_iter (int): Amount of iterations.
                }

        Returns:
            coefficients (numpy.array): The final weights after gradient ascent finishes.

        """
        # Make sure we are using numpy array
        coefficients = np.array(model_parameters["initial_coefficients"])

        # set seed=1 to produce consistent results
        np.random.seed(seed=1)

        # Shuffle the data before starting
        permutation = np.random.permutation(len(feature_matrix))
        feature_matrix = feature_matrix[permutation, :]
        label = label[permutation]

        # index of current batch
        i = 0

        # Do a linear scan over data
        for _ in range(model_parameters["max_iter"]):
            # we would need to compute the prediction, which is based on the link function, and we would
            # slice the i-th row of feature_matrix with [i:i+batch_size, :], this will give us rows between
            # i and i+batch size
            #           1
            # -------------------   = P(y=1|x_i,w)
            # 1 + exp(-w^t*h(x_i))
            dot_product_results = np.apply_along_axis(lambda feature, coef: np.dot(np.transpose(coef), feature),
                                                      1, feature_matrix[i: i + model_parameters["batch_size"], :],
                                                      coefficients)

            # Compute P(y_i = +1 | x_i, w) using the link function
            predictions = [1 / (1 + math.exp(-weight_dot_feature)) for weight_dot_feature in dot_product_results]

            # Compute indicator value for (y_i = +1), and we would slice it with [i:i+batch_size] to give us rows
            # between i and i+batch_size
            indicator = (label[i:i + model_parameters["batch_size"]] == +1)

            # Compute the errors as indicator - predictions, (1[y=+1]-P(y=1|x_i,w)
            errors = indicator - predictions

            # Compute the coefficients by using w^(t) + n*Σ^N_b=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w)) * norm constant
            # To do a summation to batch size, we will need to slice feature_matrix from [i:i+batch_size] to give us
            # range i, i+batch_size rows
            # We do a transpose of feature matrix to convert rows into the column data, since the
            # the sigma function works on all the values for a specific column, and we will multiply each
            # row will error, which gives us Σ^N_b=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            # Then norm constant is (1/batch_size), multiplying this with the summation gives us
            # n*Σ^N_b=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w)) * norm constant
            coefficients = coefficients + model_parameters["step_size"] * np.dot(
                np.transpose(feature_matrix[i:i + model_parameters["batch_size"], :]),
                errors) * (1. / model_parameters["batch_size"])

            # i is the current index of row of feature_matrix, increment it so that we can see the next batch
            i += model_parameters["batch_size"]

            # If we made a complete pass over the data, we would need to shuffle and restart again
            if i + model_parameters["batch_size"] > len(feature_matrix):
                permutation = np.random.permutation(len(feature_matrix))
                feature_matrix = feature_matrix[permutation, :]
                label = label[permutation]
                i = 0

        return coefficients
