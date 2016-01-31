import unittest
import pandas as pd
import numpy as np
from machine_learning.regression.linear_regression.LinearRegression import LinearRegression
from data_extraction.convert_numpy.convert_numpy import ConvertNumpy

# tests_http_client.py contains tests for testing our tornado_http_client

class TestLinearRegression(unittest.TestCase):
    #   Usage:
    #       Tests for the Linear Regression Class.

    def setUp(self):
        # Usage:
        #       Constructor for TestHttpClient. Used for setting the http_client.
        # Arguments:
        #       http_client (object) : an object of our HTTPClient class

        # Create an instance of the Convert Numpy class
        self.convert_numpy = ConvertNumpy()

        # Create an instance of the Linear Regression class
        self.linear_regression = LinearRegression()

        # Create a dictionary type to store relevant data types so that our pandas
        # will read the correct information
        dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
                      'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
                      'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,
                      'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int,
                      'view':int}

        # Create a kc_house_frame that encompasses all test and train data
        self.kc_house_frame = pd.read_csv('./unit_tests/test_data/kc_house/kc_house_data.csv', dtype=dtype_dict)

        # Create a kc_house_test_frame that encompasses only train data
        self.kc_house_train_frame = pd.read_csv('./unit_tests/test_data/kc_house/kc_house_train_data.csv', dtype=dtype_dict)

        # Create a kc_house_frames that encompasses only test data
        self.kc_test_frames = pd.read_csv('./unit_tests/test_data/kc_house/kc_house_test_data.csv', dtype=dtype_dict)

    def test_01_gradient_descent(self):
        # Usage:
        #       Tests the result on gradient descent
        # Arguments:
        #       None

        # We will use sqft_living for our features
        features = ['sqft_living']

        # Output will use price
        output = ['price']

        # Convert our pandas frame to numpy
        feature_matrix, output = self.convert_numpy.convert_to_numpy(self.kc_house_train_frame, features, output, 1)

        print(feature_matrix)
        print(output)

        # Create our initial weights
        initial_weights = np.array([-47000., 1.])

        # Step size
        step_size = 7e-12

        # Tolerance
        tolerance = 2.5e7

        # Compute our gradient descent value
        final_weights = self.linear_regression.gradient_descent(feature_matrix, output,
                                                                initial_weights, step_size,
                                                                tolerance)

        # Assert that the weights is correct
        self.assertEquals(-46999.887165546708, final_weights[0])
        self.assertEquals(281.91211917520917, final_weights[1])
