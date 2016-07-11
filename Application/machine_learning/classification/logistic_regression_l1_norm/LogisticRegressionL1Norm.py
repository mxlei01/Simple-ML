import numpy as np
from performance_assessment.predict_output import PredictOutput
import math


class LogisticRegressionL1Norm:
    # Usage:
    #       Lasso Regression is based on: w_j = ro_j + delta/2  if ro_j < -delta/2
    #                                           0               if ro_j between [-delta/2,delta/2]
    #                                           ro_j - delta/2  if ro_j >  delta/2
    #
    #                                   sigma(i=1, N, h_j(x_i))(1[y_i=+1])
    #                              1 -  ------------------------------------
    #                                       sigma(i=1, N, h_j(x_i)
    #         Where ro_j = log_10(-------------------------------------------)
    #                                     -sigma(k=j, N, w_k*h_k(x_i)
    #                                   e
    #                      ---------------------------------------------------
    #                                                 h_j(x_i)
    #                                        log_10 (e        )
    #         Note that e^h_j(x_i) and
    #                   e^-sigma(k=j, N, w_k*h_k(x_i)
    #                   both needs sigma(i=1, N) since it contains i
    #       where,
    #           h_j(x_i)  : features using jth coefficient for feature at i
    #           w_j       : coefficient at j
    #           1[y=+1]   : indicator function for y=+1
    #           λ(lambda) : l2 penalty

    def __init__(self):
        # Usage:
        #       Constructor for the LassoRegression class, mainly used to set the predict output class
        # Arguments:
        #       None

        # Set the Predict Output class
        self.predict_output = PredictOutput()

    def compute_ro_j(self, feature_matrix, label, coefficients):
        # Usage:
        #       Compute ro_j
        # Arguments:
        #       feature_matrix (numpy matrix) : feature matrix
        #       label          (numpy array)  : real output (not predicted) for feature_matrix
        #       coefficients   (numpy array)  : the current coefficients
        # Return:
        #       ro             (numpy array)  : ro (or new coefficients for each feature)

        # Number of features (columns)
        feature_num = feature_matrix.shape[1]

        # Set ro to be an array that is feature_num size
        ro = np.zeros(feature_num)

        # Loop through feature
        for j in range(feature_num):

            # Compute the indicator function (1[i=+1])
            indicator = (label == +1)

            # Compute the dot product of the features and the indicator function
            # sigma(i=1, N, h_j(x_i))(1[y_i=+1])
            feature_dot_indicator = np.sum(np.dot(np.transpose(feature_matrix), indicator))

            # Compute the sum of the feature matrix
            # sigma(i=1, N, h_j(x_i)
            feature_sum = np.sum(np.transpose(feature_matrix))

            # Compute 1 - feature_dot_indicator/feature_sum
            #      sigma(i=1, N, h_j(x_i))(1[y_i=+1])
            # 1 -  ------------------------------------
            #             sigma(i=1, N, h_j(x_i)
            one_minus_division = 1.0 - (feature_dot_indicator/feature_sum)

            # Get the euler power by negative weight of weight * features without current j
            #  -sigma(k=j, N, w_k*h_k(x_i)
            # e
            exponent_neg_output = np.power(math.e,
                                           -np.sum(self.predict_output.predict_output_classification(np.delete(feature_matrix, j, axis=1),
                                                                                                     np.delete(coefficients, j))))

            # Compute the logarithmic base 10 of one_minus_division and exponent_neg_output
            #               sigma(i=1, N, h_j(x_i))(1[y_i=+1])
            #         1 -  ------------------------------------
            #                sigma(i=1, N, h_j(x_i)
            # log_10(-------------------------------------------)
            #               -sigma(k=j, N, w_k*h_k(x_i)
            #              e
            log_base_10_top = math.log(one_minus_division/exponent_neg_output, 10)

            # Compute euler power of feature for current j
            #  h_j(x_i)
            # e
            euler_feature = np.power(math.e, np.sum(np.transpose(feature_matrix[:, j])))

            # Compute log of euler_feature
            log_base_10_euler = math.log(euler_feature, 10)

            # Compute the final result
            #                          sigma(i=1, N, h_j(x_i))(1[y_i=+1])
            #                     1 -  ------------------------------------
            #                                sigma(i=1, N, h_j(x_i)
            # Where ro_j = log_10(-------------------------------------------)
            #                                -sigma(k=j, N, w_k*h_k(x_i)
            #                               e
            #              ---------------------------------------------------
            #                                  h_j(x_i)
            #                         log_10 (e        )
            ro[j] = log_base_10_top/log_base_10_euler

        return ro

    def lasso_coordinate_descent_step(self, i, feature_matrix, label, coefficients, l1_penalty):
        # Usage:
        #       Computes the lasso coordinate descent step for coefficient
        # Arguments:
        #       i              (int)          : feature i
        #       feature_matrix (numpy matrix) : feature matrix
        #       label          (numpy array)  : real label for feature_matrix
        #       coefficients   (numpy array)  : current coefficients
        #       l1_penalty     (double)       : l1 penalty value
        # Return:
        #       new_coefficient_i   (double)       : new coefficient for the feature

        # compute ro[i] = SUM[ [feature_i]*(output - prediction + coefficient[i]*[feature_i]) ]
        ro_i = self.compute_ro_j(feature_matrix, label, coefficients)[i]

        # when i == 0, then it's a intercept -- do not regularize
        # else
        #   w_j = ro_j + delta/2  if ro_j < -delta/2
        #         0               if ro_j between [-delta/2,delta/2]
        #         ro_j - delta/2  if ro_j >  delta/2
        if i == 0:
            new_coefficient_i = ro_i
        elif ro_i < -l1_penalty/2.:
            new_coefficient_i = ro_i + l1_penalty/2
        elif ro_i > l1_penalty/2.:
            new_coefficient_i = ro_i - l1_penalty/2
        else:
            new_coefficient_i = 0.

        # Return the new coefficient for feature i
        return new_coefficient_i

    def lasso_cyclical_coordinate_descent(self, feature_matrix, label, initial_coefficients, l1_penalty, tolerance):
        # Usage:
        #       Performs a Lasso Cyclical Coordinate Descent, which will loop over each features and then perform
        #       coordinate descent, and if all of the coefficient changes are less than the tolerance, then we will
        #       stop.
        # Arguments:
        #       feature_matrix       (numpy matrix) : feature matrix
        #       label                (numpy array)  : real output for the feature matrix
        #       initial_coefficients (numpy array)  : the starting initial coefficients
        #       l1_penalty           (double)       : l1 penalty value
        #       tolerance            (double)       : tolerance to test against all changed coefficients
        # Return:
        #       coefficients         (numpy array)  : final coefficients after coordinate descent has been completed

        # Flag to indicate that the change is too low
        low_change = False

        # Set coefficients to initial_coefficients
        coefficients = initial_coefficients

        # While the change is not too low (meaning lower than tolerance)
        while not low_change:

            # An array of boolean to detect if all the changes are less than tolerance
            change = []

            # Need to incorporate all the new changes to the coefficients
            for i in range(len(coefficients)):
                # Remember the old coefficients
                old_coefficients_i = coefficients[i]

                # Compute the current coefficient
                coefficients[i] = self.lasso_coordinate_descent_step(i,
                                                                     feature_matrix,
                                                                     label,
                                                                     coefficients,
                                                                     l1_penalty)

                # Returns true if any coefficient changes greater than tolerance
                change.append(abs(old_coefficients_i-coefficients[i]) > tolerance)

            # Returns true if all the changes are less than tolerance
            low_change = not any(change)

        return coefficients
