"""Implements TestWeightedBinaryDecisionTrees Unittest."""

import unittest
import pandas as pd
from performance_assessment.predict_output import PredictOutput
from performance_assessment.accuracy import Accuracy
from performance_assessment.error import Error
from machine_learning.classification.weighted_binary_decision_trees import WeightedBinaryDecisionTrees
from machine_learning.ensembles.adaboost import AdaBoost


class TestWeightedBinaryDecisionTrees(unittest.TestCase):

    """Tests for the BinaryDecisionTrees class.

    Uses lending club data to test binary decision trees.

    Statics:
        _multiprocess_can_split_ (bool): Flag for nose tests to run tests in parallel.

    """

    _multiprocess_can_split_ = True

    def setUp(self):
        """Constructor for WeightedBinaryDecisionTrees.

        We will clean up the loans_data by doing one hot encoding our features list, however, in the end we will
        use some pre-built data for training and testing, but it uses the same method for one hot encoding.

        """
        self.weighted_binary_decision_trees = WeightedBinaryDecisionTrees()
        self.adaboost = AdaBoost()
        self.predict = PredictOutput()
        self.accuracy = Accuracy()
        self.error = Error()

        # Pandas type set
        dtype_dict = {'grade': str, 'term': str, 'emp_length': str, 'bad_loans': int}

        # Load the lending club data
        self.loans_data = pd.read_csv('./unit_tests/test_data/classification/lending_club/lending_club_data.csv',
                                      dtype=dtype_dict)

        # List features and targets that we are interested
        self.features = ['grade', 'term', 'home_ownership', 'emp_length']
        self.target = 'safe_loans'

        # Do a one hot encoding of features
        for feature in self.features:
            # One hot encode
            loans_data_one_hot_encoded = pd.get_dummies(self.loans_data[feature].apply(lambda x: x),
                                                        prefix=feature, prefix_sep='.')

            # Drop the feature
            self.loans_data.drop(feature, axis=1, inplace=True)

            # Join the feature with the new one encoded features
            self.loans_data = pd.concat([self.loans_data, loans_data_one_hot_encoded], axis=1)

        # Update our features
        self.features = list(self.loans_data.columns.values)
        self.features.remove('safe_loans')

        # Load our training and testing data
        self.train_data = pd.read_csv('./unit_tests/test_data/classification/lending_club/lending_club_train.csv')
        self.test_data = pd.read_csv('./unit_tests/test_data/classification/lending_club/lending_club_test.csv')

    def test_01_greedy_recursive(self):
        """Tests greedy recursive function for BinaryDecisionTrees class

        We will use the training data to build a decision tree, and measure the accuracy with some known good values.

        """
        # Create data weights
        data_weights = pd.Series([1.] * 10 + [0.] * (len(self.train_data) - 20) + [1.] * 10)

        # Create a decision tree
        decision_tree = self.weighted_binary_decision_trees.greedy_recursive(self.train_data, self.features,
                                                                             self.target, data_weights,
                                                                             current_depth=0, max_depth=2)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(decision_tree, self.train_data, self.target)

        # Assert that the classification should be 0.48124865678057166
        self.assertEqual(round(accuracy, 5), round(0.48124865678057166, 5))

    def test_02_greedy_recursive_high_depth_low_features(self):
        """Tests greedy recursive function for BinaryDecisionTrees class

        We will use the training data to build a decision tree, and measure the accuracy with some known good values.

        """
        # Create data weights
        data_weights = pd.Series([1.] * 10 + [0.] * (len(self.train_data) - 20) + [1.] * 10)

        # Create a decision tree
        decision_tree = self.weighted_binary_decision_trees.greedy_recursive(self.train_data, ['grade.A', 'grade.B'],
                                                                             self.target, data_weights,
                                                                             current_depth=0, max_depth=2000)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(decision_tree, self.train_data, self.target)

        # Assert that the classification should be 0.54148
        self.assertEqual(round(accuracy, 5), round(0.54148, 5))

    def test_03_greedy_recursive_high_depth_low_features(self):
        """Tests greedy recursive function for BinaryDecisionTrees class

        We will use the training data to build a decision tree, and measure the accuracy with some known good values.

        """
        # Create data weights
        data_weights = pd.Series([1.] * 10 + [2.] * (len(self.train_data) - 20) + [-1.] * 10)

        # Create a decision tree
        decision_tree = self.weighted_binary_decision_trees.greedy_recursive(self.train_data, ['grade.A', 'grade.B',
                                                                                               'grade.C', 'grade.D',
                                                                                               'grade.E', 'grade.F',
                                                                                               'grade.G',
                                                                                               'term. 36 months'],
                                                                             self.target, data_weights,
                                                                             current_depth=0, max_depth=20000)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(decision_tree, self.train_data, self.target)

        # Assert that the classification should be 0.38491
        self.assertEqual(round(accuracy, 5), round(0.38491, 5))

    def test_04_adaboost(self):
        """Tests the adaboost algorithm.

        Tests the adaboost algorithm with low number of iterations.

        """
        # Create two weighted binary decision trees
        weights_list, _ = self.adaboost.decision_tree(self.train_data, self.features, self.target,
                                                      iterations=2,
                                                      predict_method=self.predict.binary_tree,
                                                      model=self.weighted_binary_decision_trees,
                                                      model_method="greedy_recursive",
                                                      model_parameters={"max_depth": 1})

        # The weights have to equal to [0.15802933659263743, 0.1768236329364191]
        self.assertEqual([round(i, 5) for i in weights_list],
                         [round(0.15802933659263743, 5), round(0.1768236329364191, 5)])

    def test_05_adaboost_high_iterations(self):
        """Tests the adaboost algorithm.

        Tests the adaboost algorithm with high number of iterations.

        """
        # Create ten weighted binary decision trees
        weights_list, models_list = self.adaboost.decision_tree(self.train_data, self.features, self.target,
                                                                iterations=10,
                                                                predict_method=self.predict.binary_tree,
                                                                model=self.weighted_binary_decision_trees,
                                                                model_method="greedy_recursive",
                                                                model_parameters={"max_depth": 1})

        # Get the predictions of each dataset in the test data
        predictions = self.predict.adaboost_binary_decision_tree(self.predict.binary_tree, models_list, weights_list,
                                                                 self.test_data)

        # Assert the predictions
        self.assertEqual(list(predictions)[0:20],
                         [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])

        # Accuracy has to match 0.620314519604
        self.assertEqual(round(self.accuracy.decision_tree(self.test_data,
                                                           predictions,
                                                           self.target),
                               5),
                         round(0.620314519604, 5))
