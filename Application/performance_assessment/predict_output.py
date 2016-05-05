import numpy as np

class PredictOutput:
    # Usage:
    #   Used for predicting outputs based on features and weights

    def predict_output_regression(self, feature_matrix, weights):
        # Usage:
        #   Predicts output based on y_hat = Hw
        # Arguments:
        #   feature_matrix (numpy array) : a numpy array containing features
        #   weights        (numpy array) : a numpy array containing weights]]
        # Return:
        #   Hw

        return np.dot(feature_matrix, weights)
