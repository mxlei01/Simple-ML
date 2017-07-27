"""Implements TestBinaryDecisionTrees Unittest."""

import unittest
import pandas as pd
from performance_assessment.predict_output import PredictOutput
from performance_assessment.accuracy import Accuracy
from performance_assessment.error import Error
from machine_learning.classification.binary_decision_trees import BinaryDecisionTrees


class TestBinaryDecisionTrees(unittest.TestCase):

    """Tests for the BinaryDecisionTrees class.

    Uses lending club data to test binary decision trees.

    Attributes:
        binary_decision_trees (BinaryDecisionTrees): Binary decision tree class.
        predict_output (PredictOutput): Predict output class to predict output for decision tree.
        accuracy (Accuracy): Measures the accuracy of algorithms.
        error (Error): Measure the accuracy of algorithms.
        loans_data (pandas.DataFrame): Lending Club Data.
        features (list of str): List of features to build decision tree on.
        target (str): The target that we are predicting.
        train_data (pandas.DataFrame): Lending Club training Data.
        test_data (pandas.DataFrame): Lending Club testing data.

    Statics:
        _multiprocess_can_split_ (bool): Flag for nose tests to run tests in parallel.

    """

    _multiprocess_can_split_ = True

    def setUp(self):
        """Set up for TestBinaryDecisionTrees.

        We will clean up the loans_data by doing one hot encoding our features list, however, in the end we will
        use some pre-built data for training and testing, but it uses the same method for one hot encoding.

        """
        self.binary_decision_trees = BinaryDecisionTrees()
        self.predict_output = PredictOutput()
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
        """Tests greedy recursive function for BinaryDecisionTrees class.

        We will use the training data to build a decision tree, and measure the accuracy with some known good values.

        """
        # Create a decision tree
        decision_tree = self.binary_decision_trees.greedy_recursive(self.train_data, self.features, self.target,
                                                                    {"current_depth": 0, "max_depth": 6})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(decision_tree, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(decision_tree, self.test_data, self.target)

        # Assert that the classification should be 0.3837785437311504
        self.assertEqual(round(accuracy, 5), round(0.3837785437311504, 5))

    def test_02_greedy_recursive_high_depth_low_feature(self):
        """Tests greedy recursive function for BinaryDecisionTrees class.

        We will use the training data to build a decision tree, and use high depth.

        """
        # Create a decision tree
        decision_tree = self.binary_decision_trees.greedy_recursive(self.train_data, ['grade.A', 'grade.B'],
                                                                    self.target, {"current_depth": 0,
                                                                                  "max_depth": 1000})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(decision_tree, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(decision_tree, self.test_data, self.target)

        # Assert that the classification should be 0.38432
        self.assertEqual(round(accuracy, 5), round(0.38432, 5))

    def test_04_greedy_recursive_high_depth(self):
        """Tests greedy recursive function for BinaryDecisionTrees class.

        We will use the training data to build a decision tree, and use high depth.

        """
        # Create a decision tree
        decision_tree = self.binary_decision_trees.greedy_recursive(self.train_data, self.features, self.target,
                                                                    {"current_depth": 0, "max_depth": 10000})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(decision_tree, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(decision_tree, self.test_data, self.target)

        # Assert that the classification should be 0.37732
        self.assertEqual(round(accuracy, 5), round(0.37732, 5))

    def test_03_greedy_recursive_early_stop(self):
        """Tests for greedy recursive with early stopping for BinaryDecisionTrees class.

        We will use early stopping for greedy recursive, and measure performance.

        """
        # Create a model with max_depth=6, min_node_size=100, min_error_reduction=0
        model_1 = self.binary_decision_trees.greedy_recursive_early_stop(self.train_data, self.features, self.target,
                                                                         {"current_depth": 0, "max_depth": 6,
                                                                          "min_node_size": 100,
                                                                          "min_error_reduction": 0.0})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(model_1, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(model_1, self.test_data, self.target)

        # Assert that the classification should be 0.38367083153813014
        self.assertEqual(round(accuracy, 5), round(0.38367083153813014, 5))

        # Create a model with max_depth=6, min_node_size=0, min_error_reduction=-1
        model_2 = self.binary_decision_trees.greedy_recursive_early_stop(self.train_data, self.features, self.target,
                                                                         {"current_depth": 0, "max_depth": 6,
                                                                          "min_node_size": 0,
                                                                          "min_error_reduction": -1})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(model_2, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(model_2, self.test_data, self.target)

        # Assert that the classification should be 0.3837785437311504
        self.assertEqual(round(accuracy, 5), round(0.3837785437311504, 5))

    def test_04_greedy_recursive_early_stop_high_depth(self):
        """Tests for greedy recursive with early stopping for BinaryDecisionTrees class.

        We will use early stopping for greedy recursive, and measure performance.

        """
        # Create a model with max_depth=5000, min_node_size=0, min_error_reduction=0
        model_1 = self.binary_decision_trees.greedy_recursive_early_stop(self.train_data, ['grade.A', 'grade.B'],
                                                                         self.target,
                                                                         {"current_depth": 0, "max_depth": 5000,
                                                                          "min_node_size": 0,
                                                                          "min_error_reduction": 0.0})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(model_1, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(model_1, self.test_data, self.target)

        # Assert that the classification should be 0.38432
        self.assertEqual(round(accuracy, 5), round(0.38432, 5))

    def test_05_greedy_recursive_early_stop_high_depth(self):
        """Tests for greedy recursive with early stopping for BinaryDecisionTrees class.

        We will use early stopping for greedy recursive, and measure performance.

        """
        # Create a model with max_depth=5000, min_node_size=0, min_error_reduction=0
        model_1 = self.binary_decision_trees.greedy_recursive_early_stop(self.train_data, ['grade.A', 'grade.B',
                                                                                           'grade.C', 'grade.D',
                                                                                           'grade.E', 'grade.F',
                                                                                           'grade.G',
                                                                                           'term. 36 months'],
                                                                         self.target,
                                                                         {"current_depth": 0, "max_depth": 5000,
                                                                          "min_node_size": 0,
                                                                          "min_error_reduction": -50000})

        # Get the classification result of the first row
        classification = self.predict_output.binary_tree(model_1, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.error.binary_tree(model_1, self.test_data, self.target)

        # Assert that the classification should be 0.38162
        self.assertEqual(round(accuracy, 5), round(0.38162, 5))
