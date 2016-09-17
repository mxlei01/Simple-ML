import numpy as np
import math


class LogisticRegression:
    # Usage:
    #   Logistic Regression with gradient ascent based on:
    #           w^(t+1) <= w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
    #       w(t)         : weight at iteration t
    #       w(t+1)       : weight at iteration t+1
    #       n            : step size
    #       h_j(X_i)     : feature for each row of a specific column
    #       1[y=+1]      : indicator function for y=+1
    #       P(y=1|x_i,w) : probability of y=1 for x_i using the current weights
    #   Logistic Regression with stochastic gradient ascent is based on:
    #           w^(t+1) <= w^(t) + n*Σ^N_b=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
    #       Σ^N_b        : summation up to batch size b, 1 for SGA, and anything above 1 is BGA (Batch)
    #       w(t)         : weight at iteration t
    #       w(t+1)       : weight at iteration t+1
    #       n            : step size
    #       h_j(X_i)     : feature for each row of a specific column
    #       1[y=+1]      : indicator function for y=+1
    #       P(y=1|x_i,w) : probability of y=1 for x_i using the current weights

    def gradient_ascent(self, feature_matrix, label, initial_coefficients, step_size, max_iter):
        # Usage:
        #       Gradient ascent algorithm: w^(t+1) <= w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       label           (numpy array)  : the label of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
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

            # Compute the coefficients by using w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            # in matrix form
            # We do a transpose of feature matrix to convert rows into the column data, since the
            # the sigma function works on all the values for a specific column, and we will multiply each
            # row will error, which gives us Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
            coefficients = coefficients+step_size*np.dot(np.transpose(feature_matrix), errors)

        return coefficients


    def stochastic_gradient_ascent(self, feature_matrix, label, initial_coefficients, step_size, batch_size,
                                   max_iter):
        # Usage:
        #       Gradient ascent algorithm: w^(t+1) <= w^(t) + n*Σ^N_i=1(h_j(X_i))(1[y=+1]-P(y=1|x_i,w))
        # Arguments:
        #       feature_matrix  (numpy matrix) : features of a dataset
        #       label           (numpy array)  : the label of a dataset
        #       initial_weights (numpy array)  : initial weights that are used
        #       step_size       (int)          : step size
        #       batch_size      (int)          : number of items to sum per iteration
        #       max_iter        (int)          : amount of iterations
        # Return:
        #       weights         (numpy array)  : the final weights after gradient descent

        # Make sure we are using numpy array
        coefficients = np.array(initial_coefficients)

        # set seed=1 to produce consistent results
        np.random.seed(seed=1)

        # Shuffle the data before starting
        permutation = np.random.permutation(len(feature_matrix))
        feature_matrix = feature_matrix[permutation, :]
        label = label[permutation]

        # index of current batch
        i = 0

        # Do a linear scan over data
        for itr in range(max_iter):
            # we would need to compute the prediction, which is based on the link function, and we would
            # slice the i-th row of feature_matrix with [i:i+batch_size, :], this will give us rows between
            # i and i+batch size
            #           1
            # -------------------   = P(y=1|x_i,w)
            # 1 + exp(-w^t*h(x_i))
            dot_product_results = np.apply_along_axis(lambda feature, coef: np.dot(np.transpose(coef), feature),
                                                      1, feature_matrix[i:i+batch_size, :], coefficients)

            # Compute P(y_i = +1 | x_i, w) using the link function
            predictions = [1/(1+math.exp(-weight_dot_feature)) for weight_dot_feature in dot_product_results]


            # Compute indicator value for (y_i = +1), and we would slice it with [i:i+batch_size] to give us rows
            # between i and i+batch_size
            indicator = (label[i:i+batch_size] == +1)

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
            coefficients = coefficients + step_size * np.dot(np.transpose(feature_matrix[i:i+batch_size,:]), errors) \
                                          * (1./batch_size)

            # i is the current index of row of feature_matrix, increment it so that we can see the next batch
            i += batch_size

            # If we made a complete pass over the data, we would need to shuffle and restart again
            if i + batch_size > len(feature_matrix):
                permutation = np.random.permutation(len(feature_matrix))
                feature_matrix = feature_matrix[permutation, :]
                label = label[permutation]
                i = 0

        return coefficients
