import numpy as np
from performance_assessment.predict_output import PredictOutput


class LassoRegression:
    # Usage:
    #       Lasso Regression is based on: w_j = ro_j + delta/2  if ro_j < -delta/2
    #                                           0               if ro_j between [-delta/2,delta/2]
    #                                           ro_j - delta/2  if ro_j >  delta/2
    #       Where ro_j = Zigma(N, i=1, h_j(x_i)(y_i-y^_i(w_-j)
    #           h_j(x_i)   : normalized features of x_i (input features, but without j feature)
    #           y_i        : real output
    #           y^_i(w_-j) : predicted output without feature j

    def __init__(self):
        # Usage:
        #       Constructor for the LassoRegression class, mainly used to set the predict output class
        # Arguments:
        #       None

        # Set the Predict Output class
        self.predict_output = PredictOutput()

    def compute_ro_j(self, feature_matrix, real_output, weights):
        # Usage:
        #       Compute ro_j
        # Arguments:
        #       feature_matrix (numpy matrix) : feature matrix
        #       real_output    (numpy array)  : real output (not predicted) for feature_matrix
        #       weights        (numpy array)  : the current weights
        # Return:
        #       ro             (numpy array)  : ro (or new weights for each feature)

        # Number of features (columns)
        feature_num = feature_matrix.shape[1]

        # Set ro to be an array that is feature_num size
        ro = np.zeros(feature_num)

        # Loop through feature
        for j in range(feature_num):

            # prediction = y_i(w_-j), prediction without feature j
            prediction = self.predict_output.predict_output_regression(np.delete(feature_matrix, j, axis=1),
                                                                       np.delete(weights, j))

            # residual = output - prediction
            residual = real_output-prediction

            # ro[j] = zigma(N, i=1, feature_i) * residual
            ro[j] = np.sum([feature_matrix[:, j]*residual])

        return ro

    def lasso_coordinate_descent_step(self, i, feature_matrix, output, weights, l1_penalty):
        # Usage:
        #       Computes the lasso coordinate descent step for weight
        # Arguments:
        #       i              (int)          : feature i
        #       feature_matrix (numpy matrix) : feature matrix
        #       output         (numpy array)  : real output for feature_matrix
        #       weights        (numpy array)  : current weights
        #       l1_penalty     (double)       : l1 penalty value
        # Return:
        #       new_weight_i   (double)       : new weight for the feature

        # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = self.compute_ro_j(feature_matrix, output, weights)[i]

        # when i == 0, then it's a intercept -- do not regularize
        # else
        #   w_j = ro_j + delta/2  if ro_j < -delta/2
        #         0               if ro_j between [-delta/2,delta/2]
        #         ro_j - delta/2  if ro_j >  delta/2
        if i == 0:
            new_weight_i = ro_i
        elif ro_i < -l1_penalty/2.:
            new_weight_i = ro_i + l1_penalty/2
        elif ro_i > l1_penalty/2.:
            new_weight_i = ro_i - l1_penalty/2
        else:
            new_weight_i = 0.

        # Return the new weight for feature i
        return new_weight_i

    def lasso_cyclical_coordinate_descent(self, feature_matrix, output, initial_weights, l1_penalty, tolerance):
        # Usage:
        #       Performs a Lasso Cyclical Coordinate Descent, which will loop over each features and then perform
        #       coordinate descent, and if all of the weight changes are less than the tolerance, then we will
        #       stop.
        # Arguments:
        #       feature_matrix  (numpy matrix) : feature matrix
        #       output          (numpy array)  : real output for the feature matrix
        #       initial_weights (numpy array)  : the starting initial weights
        #       l1_penalty      (double)       : l1 penalty value
        #       tolerance       (double)       : tolerance to test against all changed weights
        # Return:
        #       weights         (numpy array)  : final weights after coordinate design has been completed

        # Flag to indicate that the change is too low
        low_change = False

        # Set Weights to initial_weights
        weights = initial_weights

        # While the change is not too low (meaning lower than tolerance)
        while not low_change:

            # An array of boolean to detect if all the changes are less than tolerance
            change = []

            # Need to incorporate all the new changes to the weights
            for i in range(len(weights)):
                # Remember the old weights
                old_weights_i = weights[i]

                # Compute the current weight
                weights[i] = self.lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)

                # Returns true if any weight changes greater than tolerance
                change.append(abs(old_weights_i-weights[i]) > tolerance)

            # Returns true if all the changes are less than tolerance
            low_change = not any(change)

        return weights
