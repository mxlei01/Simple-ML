"""Implements WeightedLogisticRegressionL2Norm."""

import math
import numpy as np


class WeightedLogisticRegressionL2Norm:

    """Class that has various functions to compute Weighted Logistic Regression with L2 Norm.

    Weighted Logistic Regression is essentially using linear regression techniques to compute classification problems
    by fitting a line between two classes, however, we use weights so that we can train the model to focus on data
    points that we misclassified, furthermore we can use this with Adaboost ensemble learning. This algorithm uses L2
    norm to minimizes coefficients.

    """

    @staticmethod
    def gradient_ascent(feature_matrix, label, model_parameters):
        """Weighted Gradient ascent algorithm with L2 Norm for Logistic Regression.

        The gradient ascent algorithm: w^(t+1) <= w^(t) + n(Σ^N_i=1α(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))-2*λ*w(t)).
        Where:
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            α_i: Weight at each point i.
            h_j(X_i): Feature for each row of a specific column.
            1[y=+1]: Indicator function for y=+1.
            P(y=1|x_i,w): Probability of y=1 for x_i using the current weights.
            λ(lambda): L2 penalty.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            label (numpy.array): The label of a dataset.
            model_parameters (dict): A dictionary of model parameters,
                {
                    initial_coefficients (numpy.array): Initial weights for the model,
                    weights_list (numpy.array): List of weight,
                    step_size (float): Step size,
                    max_iter (int): Amount of iterations,
                    l2_penalty (float): L2 penalty value.
                }

        Returns:
            coefficients (numpy.array): The final weights after gradient ascent finishes.

        """
        # Make sure we are using numpy array
        coefficients = np.array(model_parameters["initial_coefficients"])

        # Compute the coefficients up to max_iter
        for _ in range(model_parameters["max_iter"]):
            #           1
            # -------------------   = P(y=1|x_i,w)
            # 1 + exp(-w^t*h(x_i))
            dot_products = np.apply_along_axis(lambda feature, coef: np.dot(np.transpose(coef), feature),
                                               1, feature_matrix, coefficients)

            # Compute P(y_i = +1 | x_i, w) using the link function
            predictions = [1 / (1 + math.exp(-coefficient_dot_feature)) for coefficient_dot_feature in dot_products]

            # Compute indicator value for (y_i = +1)
            indicator = (label == +1)

            # Compute the errors as indicator - predictions
            errors = indicator - predictions

            # Need to compute the intercept, because it does not have L2 normalization
            # This is based on MLE: w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            intercept = coefficients[0] + model_parameters["step_size"] * np.dot(
                np.transpose(feature_matrix[:, 0]), errors)

            # Compute the coefficients by using w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))-2*lambda*coefficients
            # in matrix form
            # We do a transpose of feature matrix to convert rows into the column data, since the
            # the sigma function works on all the values for a specific column, and we will multiply each
            # row will error, then multiply by weights, which gives us Σ^N_i=1α(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            coefficients = coefficients + model_parameters["step_size"] * (np.dot(
                model_parameters["weights_list"] * np.transpose(feature_matrix),
                errors) - 2 * model_parameters["l2_penalty"] * coefficients)

            # The first coefficient should not be affected by L2 normalization
            coefficients[0] = intercept

        # Return the coefficients
        return coefficients
