import numpy as np
import math


class WeightedLogisticRegression:
    """Class that has various functions to compute Weighted Logistic Regression.

    Weighted Logistic Regression is essentially using linear regression techniques to compute classification problems
    by fitting a line between two classes, however, we use weights so that we can train the model to focus on data
    points that we misclassified, furthermore we can use this with Adaboost ensemble learning.

    """

    @staticmethod
    def gradient_ascent(feature_matrix, label, initial_coefficients, weights_list, step_size, max_iter):
        """Weighted Gradient ascent algorithm for Logistic Regression.

        The gradient ascent algorithm: w^(t+1) <= w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w)).
        Where:
            w(t): Weight at iteration t.
            w(t+1): Weight at iteration t+1.
            n: Step size.
            α_i: Weight at each point i.
            h_j(X_i): Feature for each row of a specific column.
            1[y=+1]: Indicator function for y=+1.
            P(y=1|x_i,w): Probability of y=1 for x_i using the current weights.

        Args:
            feature_matrix (numpy.matrix): Features of a dataset.
            label (numpy.array): The label of a dataset.
            initial_coefficients (numpy.array): Initial weights for the model.
            weights_list (numpy.array): List of weights
            step_size (int): Step size.
            max_iter (int): Amount of iterations.

        Returns:
            coefficients (numpy.array): The final weights after gradient ascent finishes.

        """
        # Make sure we are using numpy array
        coefficients = np.array(initial_coefficients)

        # Compute the weights up to max_iter
        for _ in range(max_iter):
            # we would need to compute the prediction, which is based on the link function
            #           1
            # -------------------   = P(y=1|x_i,w)
            # 1 + exp(-w^t*h(x_i))
            dot_product_results = np.apply_along_axis(lambda feature, coef: np.dot(np.transpose(coef), feature),
                                                      1, feature_matrix, coefficients)

            # Compute P(y_i = +1 | x_i, w) using the link function
            predictions = [1/(1+math.exp(-weight_dot_feature)) for weight_dot_feature in dot_product_results]

            # Compute indicator value for (y_i = +1)
            indicator = (label == +1)

            # Compute the errors as indicator - predictions, (1[y=+1]-P(y=1|x_i,w)
            errors = indicator - predictions

            # Compute the coefficients by using w^(t) + n*Σ^N_i=1α_i(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            # in matrix form
            # We do a transpose of feature matrix to convert rows into the column data, since the
            # the sigma function works on all the values for a specific column, and we will multiply each
            # row will error, then multiply by weights, which gives us Σ^N_i=1α(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            coefficients = coefficients+step_size*np.dot(weights_list*np.transpose(feature_matrix), errors)

        return coefficients
