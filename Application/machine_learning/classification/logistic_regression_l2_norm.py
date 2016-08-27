import numpy as np
import math


class LogisticRegressionL2Norm:
    # Usage:
    #   Logistic Regression with L2 Norm is based on:
    #       w^(t+1) <= w^(t) + n(Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))-2*λ*w(t)),
    #   for gradient ascent,
    #       w(t)         : coefficient at iteration t
    #       w(t+1)       : coefficient at iteration t+1
    #       n            : step size
    #       h_j(X_i)     : feature for each row of a specific column
    #       1[y=+1]      : indicator function for y=+1
    #       P(y=1|x_i,w) : probability of y=1 for x_i using the current coefficients
    #       λ(lambda)       : l2 penalty

    def gradient_ascent(self, feature_matrix, label, initial_coefficients, step_size, max_iter, l2_penalty):
        # Usage:
        #       Gradient ascent algorithm: w^(t+1) <= w^(t) + n(Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))-2*λ*w(t)),
        # Arguments:
        #       feature_matrix       (numpy matrix) : features of a dataset
        #       label                (numpy array)  : the label of a dataset
        #       initial_coefficients (numpy array)  : initial coefficients that are used
        #       step_size            (int)          : step size
        #       max_iter             (int)          : amount of iterations
        #       l2_penalty           (float)        : l2 penalty value
        # Return:
        #       coefficients         (numpy array)  : the final coefficients after gradient descent

        # Make sure we are using numpy array
        coefficients = np.array(initial_coefficients)

        # Compute the coefficients up to max_iter
        for itr in range(max_iter):
            #           1
            # -------------------   = P(y=1|x_i,w)
            # 1 + exp(-w^t*h(x_i))
            dot_product_results = np.apply_along_axis(lambda feature, coef: np.dot(np.transpose(coef), feature),
                                                      1, feature_matrix, coefficients)

            # Compute P(y_i = +1 | x_i, w) using the link function
            predictions = [1/(1+math.exp(-coefficient_dot_feature)) for coefficient_dot_feature in dot_product_results]

            # Compute indicator value for (y_i = +1)
            indicator = (label == +1)

            # Compute the errors as indicator - predictions
            errors = indicator - predictions

            # Need to compute the intercept, because it does not have L2 normalization
            # This is based on MLE: w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            intercept = coefficients[0]+step_size*np.dot(np.transpose(feature_matrix[:, 0]), errors)

            # Compute the coefficients by using w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))-2*lambda*coefficients
            # in matrix form
            # We do a transpose of feature matrix to convert rows into the column data, since the
            # the sigma function works on all the values for a specific column, and we will multiply each
            # row will error, which gives us Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            coefficients = coefficients+step_size*(np.dot(np.transpose(feature_matrix), errors)-2*l2_penalty*coefficients)

            # The first coefficient should not be affected by L2 normalization
            coefficients[0] = intercept

        # Return the coefficients
        return coefficients
