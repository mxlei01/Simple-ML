import unittest
import numpy as np
import pandas as pd
from data_extraction.convert_numpy import ConvertNumpy
from machine_learning.regression.linear_regression.LinearRegression import LinearRegression
from performance_assessment.predict_output import PredictOutput
from performance_assessment.residual_sum_squares import ResidualSumSquares


class TestLinearRegression(unittest.TestCase):
    #   Usage:
    #       Tests for the Linear Regression Class.

    def setUp(self):
        # Usage:
        #       Constructor for TestLinearRegression
        # Arguments:
        #       None

        # Create an instance of the Convert Numpy class
        self.convert_numpy = ConvertNumpy()

        # Create an instance of the Linear Regression class
        self.linear_regression = LinearRegression()

        # Create an instance of the Predict Output Class
        self.predict_output = PredictOutput()

        # Create an instance of the Residual Sum Squares Class
        self.residual_sum_squares = ResidualSumSquares()

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
        self.assertEquals(round(-46999.887165546708, 3), round(final_weights[0], 3))
        self.assertEquals(round(281.91211917520917, 3), round(final_weights[1], 3))

    def test_02_gradient_descent_multiple(self):
        # Usage:
        #   Computes gradient descent on multiple input, and computes predicted model and RSS
        # Arguments:
        #   None

        # We will use sqft_iving, and sqft_living15
        features = ['sqft_living', 'sqft_living15']

        # Output will be price
        output = ['price']

        # Convert our pandas frame to numpy
        feature_matrix, output = self.convert_numpy.convert_to_numpy(self.kc_house_train_frame, features, output, 1)

        # Create our initial weights
        initial_weights = np.array([-100000., 1., 1.])

        # Step size
        step_size = 4e-12

        # Tolerance
        tolerance = 1e9

        # Compute our gradient descent value
        final_weights = self.linear_regression.gradient_descent(feature_matrix, output,
                                                                initial_weights, step_size,
                                                                tolerance)

        # We will use sqft_iving, and sqft_living15
        test_features = ['sqft_living', 'sqft_living15']

        # Output will be price
        test_output = ['price']

        # Convert our test pandas frame to numpy
        test_feature_matrix, test_output = self.convert_numpy.convert_to_numpy(self.kc_test_frames, test_features, test_output, 1)

        # Predict the output of test features
        predicted_output = self.predict_output.predict_output_regression(test_feature_matrix, final_weights)

        # Compute RSS
        rss = self.residual_sum_squares.residual_sum_squares_regression(test_output, predicted_output)

        # Assert that rss is correct
        self.assertEquals(round(270263443629803.41, -3), round(rss, -3))


    def test_03_hill_climbing(self):
        # Usage:
        #       Tests the result on hill climbing
        # Arguments:
        #       None

        # We will use sqft_living for our features
        features = ['sqft_living']

        # Output will be price
        output = ['price']

        # Convert our pandas frame to numpy
        feature_matrix, output = self.convert_numpy.convert_to_numpy(self.kc_house_train_frame, features, output, 1)

        # Create our initial weights
        initial_weights = np.array([-47000., 1.])

        # Step size
        step_size = 7e-12

        # Tolerance
        tolerance = 2.5e7

        # Compute our hill climbing value
        final_weights = self.linear_regression.hill_climbing(feature_matrix, output,
                                                             initial_weights, step_size,
                                                             tolerance)

        # Assert that the weights is correct
        self.assertEquals(round(-47000.142201335177, 3), round(final_weights[0], 3))
        self.assertEquals(round(-352.86068692252599, 3), round(final_weights[1], 3))
