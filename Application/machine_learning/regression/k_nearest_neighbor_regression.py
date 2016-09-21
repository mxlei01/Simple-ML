import numpy as np
from ml_math.euclidean_distance import EuclideanDistance


class KNearestNeighborRegression:
    """Class that uses KNN to compute Regression.

    This class uses K Nearest Neighbor to compute regression by utilizing Euclidean Distance.

    Attributes:
        euclidean_distance (EuclideanDistance): EuclideanDistance class that provide a function to compute euclidean
            distance.

    """

    def __init__(self):
        """Constructor for KNearestNeighborRegression to setup EuclideanDistance.

        Constructor for the KNN class, mainly used to setup EuclideanDistance.

        """
        self.euclidean_distance = EuclideanDistance()

    def k_nearest_neighbor_regression(self, k, feature_matrix_training, feature_vector_query):
        """Computes K Nearest neighbor Regression.

        Computes KNN by computing euclidean distance for each query point, and returning the closest points.

        Args:
            k (int): Amount of neighbors.
            feature_matrix_training (numpy.matrix): A matrix of training points.
            feature_vector_query (numpy.array): Query point array.

        Returns:
            numpy.array: Indices of the feature_matrix_training that is closest to feature_vector_query in sorted order.

        """
        # Compute the euclidean distance on each item, and since the return from euclidean distance is an
        # array which we can use to sort based on the value in ascending order. This will give us indices of
        # feature_matrix_training.
        return np.argsort(self.euclidean_distance.euclidean_distance_cmp_one_value(feature_matrix_training,
                                                                                   feature_vector_query))[0:k]

    def predict_k_nearest_neighbor_regression(self, k, feature_matrix_training, output_train, feature_vector_query):
        """Predict KNN output by taking average of K points.

        Predicts the output of the k_nearest_neighbor_regression by taking the mean of the result from
        output where we get the indices from nearest knn.

        Args:
            k (int): Amount of neighbors.
            feature_matrix_training (numpy.matrix) : A matrix of training points.
            feature_vector_query (numpy.array): Query point array.
            output_train (numpy.array): Outputs for training data.

        Returns:
            float: Average value of the knn returned indexes.

        """
        # Compute the knn, then use the indices with the output to get the predicted values, and then
        # perform mean for all columns (axis=0)
        return np.mean(output_train[self.k_nearest_neighbor_regression(k, feature_matrix_training,
                                                                       feature_vector_query)],
                       axis=0)

    def predict_k_nearest_neighbor_all_regression(self, k, feature_matrix_training, output_train,
                                                  feature_matrix_query_set):
        """Predicts KNN output for each query set.

        Predicts the output of multiple k_nearest_neighbor_regression by using the predict_k_nearest_neighbor_regression
        function. Each row of the feature_matrix_query_set is computed for mean.

        Args:
            k (int): Amount of neighbors.
            feature_matrix_training (numpy.matrix): A matrix of training points.
            feature_matrix_query_set (list of float) : A list of query points.
            output_train (numpy.array): Outputs for training data.

        Returns:
            k_nn_predict_multiple (list): List of average value of the output using k_nn_indices that corresponds to
                each query point.

        """
        # For each feature_matrix_query_set which are rows of query point, compute the average knn value\
        # then return a list
        return [self.predict_k_nearest_neighbor_regression(k, feature_matrix_training, output_train, vector_query)
                for vector_query in feature_matrix_query_set]
