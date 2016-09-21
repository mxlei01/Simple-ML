import sys
from performance_assessment.residual_sum_squares import ResidualSumSquares


class DetermineKKnn:
    """Computes the best K for KNN algorithms.

    Computes the best K for KNN algorithms by finding the best K that has the lowest RSS.

    Attributes:
        residual_sum_squares (ResidualSumSquares): Class to compute residual sum of squares.

    """

    def __init__(self):
        """Constructor for DetermineKKnn to setup RSS class.

        Constructor to setup RSS Class.

        """
        self.residual_sum_squares = ResidualSumSquares()

    def determine_k_knn(self, knn_model, start_k, end_k, features_train, features_valid, output_train, output_valid):
        """Determines the best K value for knn algorithms.

        The best K value is computed by computing the lowest RSS value between K values start_k and end_k.

        Args:
            knn_model (function): A function that can be called to compute knn with features_train, output_train, and
                features_valid.
            start_k (int): Starting k value to compute.
            end_k (int): Ending k value to compute.
            features_train (numpy.matrix): A matrix of training points.
            features_valid (numpy.matrix): A matrix of validation points.
            output_train  (numpy.array): Outputs for training data.
            output_valid  (numpy.array): Outputs for validation data.

        Returns:
            A tuple of lowest_k and lowest_k_index:
                (
                    lowest_k (float): Best k value's RSS.
                    lowest_k_index (int): Best k value.
                )

        """
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
        return lowest_k, lowest_k_index
