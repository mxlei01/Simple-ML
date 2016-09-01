import numpy as np
import math


class WeightedLogisticRegression:
    # Usage:
    #   Weighted Logistic Regression is based on: w^(t+1) <= w^(t) + n*Σ^N_i=1α_i(h_j(X_i))(1[y=+1]-P(y=1|x_i,w)),
    #   for gradient ascent,
    #       w(t)         : weight at iteration t
    #       w(t+1)       : weight at iteration t+1
    #       n            : step size
    #       α_i          : weight at each point i
    #       h_j(X_i)     : feature for each row of a specific column
    #       1[y=+1]      : indicator function for y=+1
    #       P(y=1|x_i,w) : probability of y=1 for x_i using the current weights

    def gradient_ascent(self, feature_matrix, label, initial_coefficients, weights_list, step_size, max_iter):
        # Usage:
        #       Gradient ascent algorithm with weights: w^(t+1) <= w^(t) + n*Σ^N_i=1α(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       label           (numpy array)  : the label of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
        #       weights_list    (numpy array)  : list of weights
        #       step_size       (int)          : step size
        #       max_iter        (int)          : amount of iterations
        # Return:
        #       weights         (numpy array)  : the final weights after gradient descent

        # Make sure we are using numpy array
        coefficients = np.array(initial_coefficients)

        # Compute the weights up to max_iter
        for itr in range(max_iter):
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

        # Return the weights
        return coefficients
