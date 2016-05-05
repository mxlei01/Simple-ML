import numpy as np

class EuclideanDistance:
    # Usage:
    #   A class to compute euclidean distances

    def euclidean_distance(self, vector_one, vector_two):
        # Usage:
        #   Computes euclidean distances for two vectors
        #   Euclidean distance: sqrt((q1-p1)^2+(q2-p2)^2+(q3-p3)^2)
        # Arguments:
        #   vector_one         (numpy array) : first vector
        #   vector_two         (numpy array) : second vector
        # Return:
        #   euclidean_distance (double) : euclidean distance between two vectors

        # For both arrays, subtract and square element wise, and take the sum, then do a square root
        return np.sqrt(np.sum((vector_one-vector_two)**2))

    def euclidean_distance_cmp_one_value(self, feature_matrix_training, feature_vector_query):
        # Usage:
        #   Computes euclidean distances from the feature vector (query) to a matrix
        # Arguments:
        #   feature_matrix_training (numpy matrix) : the training set (or comparison we are going to make to)
        #   feature_vector_query    (numpy array)  : query point array
        # Return:
        #   euclidean_distances     (numpy array)  : an array of euclidean distances

        # For each array inside feature_matrix_training, we subtract and square
        # from feature_vector_query, and add together, which forms a matrix with multiple rows that only
        # has one value. Then we take the square root for each row (axis=1)
        return np.sqrt(np.sum((feature_matrix_training-feature_vector_query)**2, axis=1))