import unittest
import pandas as pd
import numpy as np
from machine_learning.regression.ridge_regression.RidgeRegression import RidgeRegression
from data_extraction.convert_numpy.convert_numpy import ConvertNumpy
from performance_assessment.predict_output import PredictOutput
from performance_assessment.residual_sum_squares import ResidualSumSquares
from performance_assessment.k_fold_cross_validation import KFoldCrossValidation

class TestRidgeRegression(unittest.TestCase):
    #   Usage:
    #       Tests for the Linear Regression Class.

    def setUp(self):
        # Usage:
        #       Constructor for TestRidgeRegression
        # Arguments:
        #       None

        # Create an instance of the Convert Numpy class
        self.convert_numpy = ConvertNumpy()

        # Create an instance of the Linear Regression class
        self.ridge_regression = RidgeRegression()

        # Create an instance of the Predict Output Class
        self.predict_output = PredictOutput()

        # Create an instance of the Residual Sum Squares Class
        self.residual_sum_squares = ResidualSumSquares()

        # Create an instance of the K Fold Cross Validation Class
        self.k_fold_cross_validation = KFoldCrossValidation()

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

        # Create a kc_house_train_valid_shuffled that encompasses both train and valid data and shuffled
        self.kc_house_train_valid_shuffled = pd.read_csv('./unit_tests/test_data/kc_house_with_validation_k_fold/wk3_kc_house_train_valid_shuffled.csv',
                                                         dtype=dtype_dict)

    def test_01_gradient_descent_no_penalty(self):
        # Usage:
        #       Tests the result on gradient descent with low penalty
        # Arguments:
        #       None

        # We will use sqft_living for our features
        features = ['sqft_living']

        # Output will use price
        output = ['price']

        # Convert our pandas frame to numpy
        feature_matrix, output = self.convert_numpy.convert_to_numpy(self.kc_house_train_frame, features, output, 1)

        # Create our initial weights
        initial_weights = np.array([0., 0.])

        # Step size
        step_size = 1e-12

        # Max Iterations to Run
        max_iterations = 1000

        # Tolerance
        tolerance = None

        # L2 Penalty
        l2_penalty = 0.0

        # Compute our gradient descent value
        final_weights = self.ridge_regression.gradient_descent(feature_matrix, output,
                                                               initial_weights, step_size,
                                                               tolerance, l2_penalty, max_iterations)

        # We will use sqft_iving, and sqft_living15
        test_features = ['sqft_living']

        # Output will be price
        test_output = ['price']

        # Convert our test pandas frame to numpy
        test_feature_matrix, test_output = self.convert_numpy.convert_to_numpy(self.kc_test_frames, test_features, test_output, 1)

        # Predict the output of test features
        predicted_output = self.predict_output.predict_output_linear_regression(test_feature_matrix, final_weights)

        # Compute RSS
        rss = self.residual_sum_squares.residual_sum_squares_linear_regression(test_output, predicted_output)

        # Assert that the weights is correct
        self.assertEquals(round(-0.16311351478746433, 5), round(final_weights[0], 5))
        self.assertEquals(263.02436896538489, final_weights[1])

        # Assert that rss is correct
        self.assertEquals(275723632153607.72, rss)

    def test_02_gradient_descent_high_penalty(self):
        # Usage:
        #       Tests the result on gradient descent with high penalty
        # Arguments:
        #       None

        # We will use sqft_living for our features
        features = ['sqft_living']

        # Output will use price
        output = ['price']

        # Convert our pandas frame to numpy
        feature_matrix, output = self.convert_numpy.convert_to_numpy(self.kc_house_train_frame, features, output, 1)

        # Create our initial weights
        initial_weights = np.array([0., 0.])

        # Step size
        step_size = 1e-12

        # Max Iterations to Run
        max_iterations = 1000

        # Tolerance
        tolerance = None

        # L2 Penalty
        l2_penalty = 1e11

        # Compute our gradient descent value
        final_weights = self.ridge_regression.gradient_descent(feature_matrix, output,
                                                               initial_weights, step_size,
                                                               tolerance, l2_penalty, max_iterations)

        # We will use sqft_iving
        test_features = ['sqft_living']

        # Output will be price
        test_output = ['price']

        # Convert our test pandas frame to numpy
        test_feature_matrix, test_output = self.convert_numpy.convert_to_numpy(self.kc_test_frames, test_features, test_output, 1)

        # Predict the output of test features

        predicted_output = self.predict_output.predict_output_linear_regression(test_feature_matrix, final_weights)

        # Compute RSS
        rss = self.residual_sum_squares.residual_sum_squares_linear_regression(test_output, predicted_output)

        # Assert that the weights is correct
        self.assertEquals(0.048718475774044, final_weights[0])
        self.assertEquals(124.57402057376679, final_weights[1])

        # Assert that rss is correct
        self.assertEquals(694654309578537.25, rss)

    def test_03_gradient_descent_multiple_high_penalty(self):
        # Usage:
        #       Tests the result on gradient descent with high penalty
        # Arguments:
        #       None

        # We will use sqft_iving, and sqft_living15
        features = ['sqft_living', 'sqft_living15']

        # Output will use price
        output = ['price']

        # Convert our pandas frame to numpy
        feature_matrix, output = self.convert_numpy.convert_to_numpy(self.kc_house_train_frame, features, output, 1)

        # Create our initial weights
        initial_weights = np.array([0.0, 0.0, 0.0])

        # Step size
        step_size = 1e-12

        # Max Iterations to Run
        max_iterations = 1000

        # Tolerance
        tolerance = None

        # L2 Penalty
        l2_penalty = 1e11

        # Compute our gradient descent value
        final_weights = self.ridge_regression.gradient_descent(feature_matrix, output,
                                                               initial_weights, step_size,
                                                               tolerance, l2_penalty, max_iterations)

        # We will use sqft_iving, and sqft_living15
        test_features = ['sqft_living', 'sqft_living15']

        # Output will be price
        test_output = ['price']

        # Convert our test pandas frame to numpy
        test_feature_matrix, test_output = self.convert_numpy.convert_to_numpy(self.kc_test_frames, test_features, test_output, 1)

        # Predict the output of test features
        predicted_output = self.predict_output.predict_output_linear_regression(test_feature_matrix, final_weights)

        # Compute RSS
        rss = self.residual_sum_squares.residual_sum_squares_linear_regression(test_output, predicted_output)

        # Assert that the weights is correct
        self.assertEquals(round(0.033601165521060711, 5), round(final_weights[0], 5))
        self.assertEquals(91.490167574878328, final_weights[1])
        self.assertEquals(78.437490333967176, final_weights[2])

        # Assert that rss is correct
        self.assertEquals(500408530236718.31, rss)

        # Look at the first predicted output
        self.assertEquals(270449.70602770313, predicted_output[0])

        # The first output should be 310000 in the test set
        self.assertEquals(310000.0, test_output[0])

    def test_04_gradient_descent_k_fold(self):
        # Usage:
        #       Tests best l2_penalty for ridge regression using gradient descent
        # Arguments:
        #       None

        # We will use sqft_living for our features
        features = ['sqft_living']

        # Output will use price
        output = ['price']

        # Create our initial weights
        initial_weights = np.array([0., 0.])

        # Step size
        step_size = 1e-12

        # Tolerance
        tolerance = None

        # Max Iterations to Run
        max_iterations = 1000

        # Number of Folds
        folds = 10

        # Store Cross Validation results
        cross_validation_results = []

        # We want to test l2 penalty values in [10^1, 10^2, 10^3, 10^4, ..., 10^11]
        for l2_penalty in np.logspace(1, 11, num=11):

            # Create a dictionary of model_parameters
            model_parameters = {'step_size': step_size,
                                'max_iteration': max_iterations,
                                'initial_weights': initial_weights,
                                'tolerance': tolerance,
                                'l2_penalty': l2_penalty}

            # Compute the cross validation results
            cross_validation = self.k_fold_cross_validation.k_fold_cross_validation(folds,
                                                                                    self.kc_house_train_frame,
                                                                                    self.ridge_regression.gradient_descent,
                                                                                    model_parameters, output, features)

            # Append it into the results
            cross_validation_results.append((l2_penalty, cross_validation))

        # Lowest Result
        lowest = sorted(cross_validation_results, key=lambda x: x[1])[0]

        # Assert True that 10000000 is the l2_penalty that gives the lowest cross validation error
        self.assertEquals(10000000.0, lowest[0])

        # Assert True that is the lowest l2_penalty
        self.assertEquals(120916225812152.84, lowest[1])

    def test_05_hill_climbing(self):
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
        initial_weights = np.array([0., 0.])

        # Step size
        step_size = 1e-12

        # Max Iterations to Run
        max_iterations = 1000

        # Tolerance
        tolerance = None

        # L2 Penalty
        l2_penalty = 0.0

        # Compute our hill climbing value
        final_weights = self.ridge_regression.hill_climbing(feature_matrix, output,
                                                            initial_weights, step_size,
                                                            tolerance, l2_penalty, max_iterations)

        # Assert that the weights is correct
        self.assertEquals(round(-7.7535764461428101e+70, -68), round(final_weights[0], -68))
        self.assertEquals(-1.9293745396177612e+74, final_weights[1])
