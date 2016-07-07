import unittest
import json
import numpy as np
import pandas as pd
from data_extraction.convert_numpy import ConvertNumpy
from machine_learning.classification.logistic_regression_l2_norm.LogisticRegressionL2Norm \
    import LogisticRegressionL2Norm
from ml_math.log_likelihood import LogLikelihood
from performance_assessment.accuracy import Accuracy


class TestLogisticRegressionL2Norm(unittest.TestCase):
    #   Usage:
    #       Tests for the Logistic Regression Class.

    def setUp(self):
        # Usage:
        #       Constructor for TestLogisticRegression
        # Arguments:
        #       None

        # Create an instance of the Convert Numpy class
        self.convert_numpy = ConvertNumpy()

        # Create an instance of log likelihood
        self.log_likelhood = LogLikelihood()

        # Create an instance of the accuracy class
        self.accuracy = Accuracy()

        # Load the important words
        self.important_words = json.load(open('./unit_tests/test_data/important_words.json', 'r'))

        # Create an instance of the Logistic Regression class
        self.logistic_regression_l2_norm = LogisticRegressionL2Norm()

        # Load the amazon baby train subset
        self.training_data = pd.read_csv('./unit_tests/test_data/amazon_baby_subset_train.csv')

        # Load the amazon baby train subset
        self.validation_data = pd.read_csv('./unit_tests/test_data/amazon_baby_subset_validation.csv')

    def test_01_gradient_ascent_no_penalty(self):
        # Usage:
        #       Test out the gradient ascent algorithm for logistic regression with l2 norm with no penalty
        # Arguments:
        #       None

        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix_train, label_train = self.convert_numpy.convert_to_numpy(self.training_data,
                                                                                features,
                                                                                output, 1)
        feature_matrix_valid, label_valid = self.convert_numpy.convert_to_numpy(self.validation_data,
                                                                                features,
                                                                                output, 1)

        # Compute the coefficients
        coefficients = self.logistic_regression_l2_norm.gradient_ascent(feature_matrix_train, label_train,
                                                                        initial_coefficients=np.zeros(194),
                                                                        step_size=5e-6, l2_penalty=0, max_iter=501)

        # Get the accuracy
        train_accuracy = self.accuracy.accuracy_classification(feature_matrix_train, label_train, coefficients)
        validation_accuracy = self.accuracy.accuracy_classification(feature_matrix_valid, label_valid, coefficients)

        # Make sure the accuraries are correct
        self.assertEqual(round(0.785156157787, 5), round(train_accuracy, 5))
        self.assertEqual(round(0.78143964149, 5), round(validation_accuracy, 5))

    def test_02_gradient_ascent_10_penalty(self):
        # Usage:
        #       Test out the gradient ascent algorithm for logistic regression with l2 norm with 10 l2 penalty
        # Arguments:
        #       None

        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix_train, label_train = self.convert_numpy.convert_to_numpy(self.training_data,
                                                                                features,
                                                                                output, 1)
        feature_matrix_valid, label_valid = self.convert_numpy.convert_to_numpy(self.validation_data,
                                                                                features,
                                                                                output, 1)

        # Compute the coefficients
        coefficients = self.logistic_regression_l2_norm.gradient_ascent(feature_matrix_train, label_train,
                                                                        initial_coefficients=np.zeros(194),
                                                                        step_size=5e-6, l2_penalty=10, max_iter=501)

        # Get the accuracy
        train_accuracy = self.accuracy.accuracy_classification(feature_matrix_train, label_train, coefficients)
        validation_accuracy = self.accuracy.accuracy_classification(feature_matrix_valid, label_valid, coefficients)

        # Make sure the accuraries are correct
        self.assertEqual(round(0.784990911452, 5), round(train_accuracy, 5))
        self.assertEqual(round(0.781719727383, 5), round(validation_accuracy, 5))

    def test_03_log_likelihood(self):
        # Usage:
        #       Test Log Likelihood with L2 Norm
        # Arguments:
        #       None

        # Generate test feature, coefficients, and label
        feature_matrix = np.array([[1., 2., 3.], [1., -1., -1]])
        coefficients = np.array([1., 3., -1.])
        label = np.array([-1, 1])

        # Compute the log likelihood
        lg = self.log_likelhood.log_likelihood_l2_norm(feature_matrix, label, coefficients, 10)

        # Assert the value
        self.assertEqual(round(lg, 5), round(-105.33141000000001, 5))
