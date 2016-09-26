"""Implements TestWeightedLogisticRegressionL2Norm Unittest."""

import json
import string
import unittest
import numpy as np
import pandas as pd
from data_extraction.convert_numpy import ConvertNumpy
from machine_learning.classification.weighted_logistic_regression_l2_norm import WeightedLogisticRegressionL2Norm
from machine_learning.ensembles.adaboost import AdaBoost
from performance_assessment.predict_output import PredictOutput
from performance_assessment.accuracy import Accuracy


class TestWeightedLogisticRegressionL2Norm(unittest.TestCase):

    """Tests WeightedLogisticRegression class.

    Uses Amazon data to test WeightedLogisticRegression class.

    """

    def setUp(self):
        """Constructor for TestWeightedLogisticRegressionL2Norm.

        Loads Amazon data, and creates training and testing data.

        """
        self.convert_numpy = ConvertNumpy()
        self.predict = PredictOutput()
        self.adaboost = AdaBoost()
        self.accuracy = Accuracy()
        self.weighted_logistic_regression_l2 = WeightedLogisticRegressionL2Norm()

        # Load the important words
        self.important_words = json.load(open('./unit_tests/test_data/classification/amazon/important_words.json', 'r'))

        # Load the amazon baby subset
        self.review_frame = pd.read_csv('./unit_tests/test_data/classification/amazon/amazon_baby_subset.csv')

        # Review needs to be text
        self.review_frame['review'].astype(str)

        # Clean up the punctuations
        self.review_frame['review_clean'] = self.review_frame.apply(
            axis=1,
            func=lambda row: str(row["review"]).translate(str.maketrans({key: None for key in string.punctuation})))

        # Remove any nan text
        self.review_frame['review_clean'] = self.review_frame.apply(
            axis=1,
            func=lambda row: '' if row["review_clean"] == "nan" else row["review_clean"])

        # Count the number of words that appears in each review, and make an indepedent column
        for word in self.important_words:
            self.review_frame[word] = self.review_frame['review_clean'].apply(lambda s, w=word: s.split().count(w))

    def test_01_gradient_ascent(self):
        """Tests gradient ascent algorithm.

        Tests gradient ascent algorithm and compare it with known values.

        """
        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix, sentiment = self.convert_numpy.convert_to_numpy(self.review_frame, features, output, 1)

        # Create weight list for training data
        weights_list = np.array([1]*len(self.review_frame))

        # Compute the coefficients
        coefficients = self.weighted_logistic_regression_l2.gradient_ascent(feature_matrix, sentiment,
                                                                            initial_coefficients=np.zeros(194),
                                                                            weights_list=weights_list,
                                                                            step_size=1e-7, l2_penalty=10,
                                                                            max_iter=30)

        # Assert the coefficients
        self.assertEqual([round(i, 5) for i in coefficients[0:20]],
                         [round(i, 5) for i in [0.00020000000000000001, 0.0014300000000000001, -0.00131,
                                                0.0068900000000000003, 0.0068500000000000002, 0.00034000000000000002,
                                                -0.0062399999999999999, -0.00059000000000000003, 0.0067099999999999998,
                                                0.0046600000000000001, 0.00042999999999999999, 0.0020300000000000001,
                                                0.0030300000000000001, -0.00332, 0.0015, -0.00011, 0.00115,
                                                -0.0021700000000000001, -0.00139, -0.0046600000000000001]])

        # Compute predictions
        predictions = self.predict.logistic_regression(feature_matrix, coefficients)

        # Accuracy has to match 0.74356999999999995
        self.assertEqual(round(self.accuracy.general(predictions, sentiment), 5),
                         round(0.74356999999999995, 5))

    def test_02_adaboost(self):
        """Tests adaboost algorithm.

        Tests the adaboost algorithm and compare it with known values.

        """
        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix, sentiment = self.convert_numpy.convert_to_numpy(self.review_frame, features, output, 1)

        # Create 15 weighted logistic regression
        weights, models = self.adaboost.logistic_regression(feature_matrix, sentiment,
                                                            iterations=5,
                                                            predict_method=self.predict.logistic_regression,
                                                            model_dict={"model": self.weighted_logistic_regression_l2,
                                                                        "model_method": "gradient_ascent",
                                                                        "model_parameters": {"step_size": 1e-7,
                                                                                             "max_iter": 5,
                                                                                             "l2_penalty": 10,
                                                                                             "initial_coefficients":
                                                                                                 np.zeros(194)}})

        # Get the predictions of each dataset in the test data
        predictions = self.predict.adaboost_logistic_regression(self.predict.logistic_regression,
                                                                models, weights, feature_matrix)

        # Assert the predictions
        self.assertEqual(list(predictions)[0:20],
                         [1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1])

        # Accuracy has to match 0.74356999999999995
        self.assertEqual(round(self.accuracy.general(predictions, sentiment), 5),
                         round(0.74146000000000001, 5))
