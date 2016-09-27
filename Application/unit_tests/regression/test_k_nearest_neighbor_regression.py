"""Implements TestKNearestNeighborRegression Unittest."""

import sys
import unittest
import numpy as np
import pandas as pd
from data_extraction.convert_numpy import ConvertNumpy
from data_extraction.normalize_features import NormalizeFeatures
from machine_learning.regression.k_nearest_neighbor_regression import KNearestNeighborRegression
from ml_math.euclidean_distance import EuclideanDistance
from performance_assessment.determine_k_knn import DetermineKKnn


class TestKNearestNeighborRegression(unittest.TestCase):

    """Tests for TestKNearestNeighborRegression.

    Uses housing data to test KNearestNeighborRegression.

    Statics:
        _multiprocess_can_split_ (bool): Flag for nose tests to run tests in parallel.

    """

    _multiprocess_can_split_ = True

    def setUp(self):
        """Constructor for TestKNearestNeighborRegression.

        Loads housing data, and creates training and testing data.

        """
        self.convert_numpy = ConvertNumpy()
        self.normalize_features = NormalizeFeatures()
        self.knn = KNearestNeighborRegression()
        self.euclidean_distance = EuclideanDistance()
        self.determine_k_knn = DetermineKKnn()

        # Create a dictionary type to store relevant data types so that our pandas
        # will read the correct information
        dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
                      'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str,
                      'long': float, 'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int,
                      'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int,
                      'view': int}

        # Create a kc_house that encompasses all test and train data
        self.kc_house = pd.read_csv('./unit_tests/test_data/regression/kc_house_knn/kc_house_data_small.csv',
                                    dtype=dtype_dict)

        # Create a kc_house_test_frame that encompasses only train data
        self.kc_house_train = pd.read_csv('./unit_tests/test_data/regression/kc_house_knn/'
                                          'kc_house_data_small_train.csv',
                                          dtype=dtype_dict)

        # Create a kc_house_frames that encompasses only test data
        self.kc_house_test = pd.read_csv('./unit_tests/test_data/regression/kc_house_knn/kc_house_data_small_test.csv',
                                         dtype=dtype_dict)

        # Create a kc_house_frames that encompasses only validation data
        self.kc_house_valid = pd.read_csv('./unit_tests/test_data/regression/kc_house_knn/kc_house_data_validation.csv',
                                          dtype=dtype_dict)

        # Convert all the frames with the floors to float type
        self.kc_house['floors'] = self.kc_house['floors'].astype(float)
        self.kc_house_train['floors'] = self.kc_house_train['floors'].astype(float)
        self.kc_house_test['floors'] = self.kc_house_test['floors'].astype(float)
        self.kc_house_valid['floors'] = self.kc_house_valid['floors'].astype(float)

        # Then back to int type
        self.kc_house['floors'] = self.kc_house['floors'].astype(int)
        self.kc_house_train['floors'] = self.kc_house_train['floors'].astype(int)
        self.kc_house_test['floors'] = self.kc_house_test['floors'].astype(int)
        self.kc_house_valid['floors'] = self.kc_house_valid['floors'].astype(int)

    def test_01_compute_euclidean_distance(self):
        """Tests Euclidean distance.

        Tests Euclidean distance and compare it with known values.

        """
        # List of features to convert to numpy
        feature_list = ['bedrooms',
                        'bathrooms',
                        'sqft_living',
                        'sqft_lot',
                        'floors',
                        'waterfront',
                        'view',
                        'condition',
                        'grade',
                        'sqft_above',
                        'sqft_basement',
                        'yr_built',
                        'yr_renovated',
                        'lat',
                        'long',
                        'sqft_living15',
                        'sqft_lot15']

        # Output to convert to numpy
        output = ['price']

        # Extract features and output for train, test, and validation set
        features_train, _ = self.convert_numpy.convert_to_numpy(self.kc_house_train, feature_list, output, 1)
        features_test, _ = self.convert_numpy.convert_to_numpy(self.kc_house_test, feature_list, output, 1)
        # features_valid, output_valid = self.convert_numpy.convert_to_numpy(self.kc_house_valid, feature_list,
        #                                                                    output, 1)

        # Normalize our training features, and then normalize the test set and valid set
        features_train, norms = self.normalize_features.l2_norm(features_train)
        features_test = features_test / norms
        # features_valid = features_valid / norms

        # Compute the euclidean distance
        distance = self.euclidean_distance.euclidean_distance(features_test[0], features_train[9])

        # Assert that both are equal
        self.assertEqual(round(distance, 3), round(0.059723593716661257, 3))

    def test_02_compute_euclidean_distance_query_point(self):
        """Tests Euclidean distance with a set of query points.

        Test to compute euclidean distance from a query point to multiple points in the training set

        """
        # List of features to convert to numpy
        feature_list = ['bedrooms',
                        'bathrooms',
                        'sqft_living',
                        'sqft_lot',
                        'floors',
                        'waterfront',
                        'view',
                        'condition',
                        'grade',
                        'sqft_above',
                        'sqft_basement',
                        'yr_built',
                        'yr_renovated',
                        'lat',
                        'long',
                        'sqft_living15',
                        'sqft_lot15']

        # Output to convert to numpy
        output = ['price']

        # Extract features and output for train, test, and validation set
        features_train, output_train = self.convert_numpy.convert_to_numpy(self.kc_house_train, feature_list, output, 1)
        features_test, _ = self.convert_numpy.convert_to_numpy(self.kc_house_test, feature_list, output, 1)
        # features_valid, output_valid = self.convert_numpy.convert_to_numpy(self.kc_house_valid, feature_list,
        #                                                                    output, 1)

        # Normalize our training features, and then normalize the test set and valid set
        features_train, norms = self.normalize_features.l2_norm(features_train)
        features_test = features_test / norms
        # features_valid = features_valid / norms

        # Determine the smallest euclidean distance set we get
        smallest = sys.maxsize
        smallest_index = 0
        for index, val in enumerate(self.euclidean_distance.euclidean_distance_cmp_one_value(features_train,
                                                                                             features_test[2])):
            if val < smallest:
                smallest = val
                smallest_index = index

        # Assert that we are getting the right prediction (for 1-NN neighbor)
        self.assertEqual(round(smallest, 8), round(0.00286049526751, 8))
        self.assertEqual(output_train[smallest_index], 249000)
        self.assertEqual(smallest_index, 382)

    def test_03_compute_knn(self):
        """Tests knn regression algorithm.

        Tests the knn algorithm and compare it with known values.

        """
        # List of features to convert to numpy
        feature_list = ['bedrooms',
                        'bathrooms',
                        'sqft_living',
                        'sqft_lot',
                        'floors',
                        'waterfront',
                        'view',
                        'condition',
                        'grade',
                        'sqft_above',
                        'sqft_basement',
                        'yr_built',
                        'yr_renovated',
                        'lat',
                        'long',
                        'sqft_living15',
                        'sqft_lot15']

        # Output to convert to numpy
        output = ['price']

        # Extract features and output for train, test, and validation set
        features_train, output_train = self.convert_numpy.convert_to_numpy(self.kc_house_train, feature_list, output, 1)
        features_test, _ = self.convert_numpy.convert_to_numpy(self.kc_house_test, feature_list, output, 1)
        # features_valid, output_valid = self.convert_numpy.convert_to_numpy(self.kc_house_valid, feature_list,
        #                                                                    output, 1)

        # Normalize our training features, and then normalize the test set and valid set
        features_train, norms = self.normalize_features.l2_norm(features_train)
        features_test = features_test / norms
        # features_valid = features_valid / norms

        # Assert that the array is the closest with the 3rd house in features_test
        self.assertTrue(np.array_equal(self.knn.k_nearest_neighbor_regression(4, features_train, features_test[2]),
                                       np.array([382, 1149, 4087, 3142])))

        # Assert that the 413987.5 is the correct prediction
        self.assertEqual(self.knn.predict_k_nearest_neighbor_regression(4, features_train,
                                                                        output_train, features_test[2]),
                         413987.5)

        # Compute the lowest predicted value
        lowest_predicted = sys.maxsize
        lowest_predicted_index = 0
        for index, val in enumerate(self.knn.predict_k_nearest_neighbor_all_regression(10, features_train,
                                                                                       output_train,
                                                                                       features_test[0:10])):
            if val < lowest_predicted:
                lowest_predicted = val
                lowest_predicted_index = index

        # Assert that the few values such as lowest predicted values and index are the one we expect
        self.assertEqual(lowest_predicted, 350032.0)
        self.assertEqual(lowest_predicted_index, 6)

    def test_03_compute_best_k(self):
        """Compute best K for KNN Regression.

        Compute best K using K Fold Cross Validation.

        """
        # List of features to convert to numpy
        feature_list = ['bedrooms',
                        'bathrooms',
                        'sqft_living',
                        'sqft_lot',
                        'floors',
                        'waterfront',
                        'view',
                        'condition',
                        'grade',
                        'sqft_above',
                        'sqft_basement',
                        'yr_built',
                        'yr_renovated',
                        'lat',
                        'long',
                        'sqft_living15',
                        'sqft_lot15']

        # Output to convert to numpy
        output = ['price']

        # Extract features and output for train, test, and validation set
        features_train, output_train = self.convert_numpy.convert_to_numpy(self.kc_house_train, feature_list, output, 1)
        # features_test, output_test = self.convert_numpy.convert_to_numpy(self.kc_house_test, feature_list,
        #                                                                  output, 1)
        features_valid, output_valid = self.convert_numpy.convert_to_numpy(self.kc_house_valid, feature_list, output, 1)

        # Normalize our training features, and then normalize the test set and valid set
        features_train, norms = self.normalize_features.l2_norm(features_train)
        # features_test = features_test / norms
        features_valid = features_valid / norms

        # Compute the lowest K and lowest K's RSS
        low_rss, low_idx = self.determine_k_knn.determine_k_knn(self.knn.predict_k_nearest_neighbor_all_regression,
                                                                1, 16, {"features_train": features_train,
                                                                        "features_valid": features_valid,
                                                                        "output_train": output_train,
                                                                        "output_valid": output_valid})

        # Assert that the lowest k and rss is correct
        self.assertEqual(round(low_rss, -13), round(6.73616787355e+13, -13))
        self.assertEqual(low_idx, 8)
