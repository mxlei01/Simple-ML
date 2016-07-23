import sys
from performance_assessment.residual_sum_squares import ResidualSumSquares

class DetermineKKnn:
    # Usage:
    #   Computes best K for Knn

    def __init__(self):
        # Usage:
        #       Constructor for DetermineKKnn, used to setup RSS computation
        #       data to numpy.
        # Arguments:
        #       None

        # Create an instance of the Residual Sum Squares Class
        self.residual_sum_squares = ResidualSumSquares()

    def determine_k_knn(self, knn_model, start_k, end_k, features_train, features_valid, output_train, output_valid):
        # Usage:
        #       Determine the best K value for knn_model
        # Arguments:
        #       knn_model      (func)         : a function that can be called to compute knn with features_train,
        #                                       output_train, and features_valid
        #       start_k        (int)          : starting k value to compute
        #       end_k          (int)          : ending k value to compute
        #       features_train (numpy matrix) : a matrix of training points
        #       features_valid (numpy matrix) : a matrix of validation points
        #       output_train   (numpy array)  : outputs for training data
        #       output_valid   (numpy array)  : outputs for validation data
        # Return:
        #       lowest_k       (int)          : best k value's RSS
        #       lowest_k_index (int)          : best k value

        # Get the largest number
        lowest_k = sys.maxsize

        # This stores the index of the lowest RSS number
        lowest_k_index = 0

        # Loop through k from start_k to end_k
        for k in range(start_k, end_k):

            # Use the knn model to compute a list of average knn
            model = knn_model(k, features_train, output_train, features_valid)

            # Compute RSS by subtracting the output valid with the model
            rss = self.residual_sum_squares.residual_sum_squares_regression(output_valid, model)

            # If the rss is less than our lowest k,
            if rss < lowest_k:

                # Update the best k value and best k's value RSS
                lowest_k = rss
                lowest_k_index = k

        # Return the best k value and it's RSS
        return (lowest_k, lowest_k_index)
