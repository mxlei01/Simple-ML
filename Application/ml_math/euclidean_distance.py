"""Implements EuclideanDistance."""

import numpy as np


class EuclideanDistance:

    """Class for computing euclidean distances.

    Computes Euclidean distances which is required for algorithms such as K Nearest Neighbor.

    """

    @staticmethod
    def euclidean_distance(vector_one, vector_two):
        """Computes Euclidean Distance.

        Computes euclidean distances for two vectors.
        Euclidean distance: sqrt((q1-p1)^2+(q2-p2)^2+(q3-p3)^2).

        Args:
            vector_one (numpy.array): First vector.
            vector_two (numpy.array): Second vector.

        Returns:
            float : Euclidean distance between two vectors.

        """
        # For both arrays, subtract and square element wise, and take the sum, then do a square root
        return np.sqrt(np.sum((vector_one-vector_two) ** 2))

    @staticmethod
    def euclidean_distance_cmp_one_value(feature_matrix_training, feature_vector_query):
        """Computes euclidean distance against a matrix.

        Computes euclidean distances from the feature vector (query) to a matrix

        Args:
            feature_matrix_training (numpy.matrix): The training set (or comparison we are going to make to).
            feature_vector_query (numpy.array): Query point array.

        Returns:
            numpy.array: An array of euclidean distances

        """
        # For each array inside feature_matrix_training, we subtract and square
        # from feature_vector_query, and add together, which forms a matrix with multiple rows that only
        # has one value. Then we take the square root for each row (axis=1)
        return np.sqrt(np.sum((feature_matrix_training-feature_vector_query) ** 2, axis=1))
