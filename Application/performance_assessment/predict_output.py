import numpy as np


class PredictOutput:
    # Usage:
    #   Used for predicting outputs based on features and weights

    def predict_output_regression(self, feature_matrix, weights):
        # Usage:
        #       Predicts output based on y_hat = Hw
        # Arguments:
        #       feature_matrix (numpy matrix) : a numpy matrix containing features
        #       weights        (numpy array)  : a numpy array containing weights
        # Return:
        #       Hw             (numpy array)

        return np.dot(feature_matrix, weights)

    def predict_output_classification(self, feature_matrix, coefficients):
        # Usage:
        #       Predicts output based on y_i = +1 hw >= 0
        #                                      -1 hw <  0
        # Arguments:
        #       feature_matrix (numpy matrix) : a numpy matrix containing features
        #       label          (numpy array)  : a numpy array containing labels
        #       coefficients   (numpy array)  : a numpy array containing coefficients
        # Return:
        #       Hw             (numpy array)

        return np.dot(feature_matrix, coefficients)
