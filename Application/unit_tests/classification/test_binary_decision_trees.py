import unittest
import pandas as pd
from performance_assessment.predict_output import PredictOutput
from performance_assessment.accuracy import Accuracy
from machine_learning.classification.binary_decision_trees import BinaryDecisionTrees


class TestBinaryDecisionTrees(unittest.TestCase):
    #   Usage:
    #       Tests for the binary decision trees class

    def setUp(self):
        # Usage:
        #       Constructor for TestLogisticRegression
        # Arguments:
        #       None

        # Create an instance of the BinaryDecisionTree class
        self.binary_decision_trees = BinaryDecisionTrees()

        # Create an instance of the Predict Output class
        self.predict_output = PredictOutput()

        # Create an instance of the accuracy class
        self.accuracy = Accuracy()

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
        # Usage:
        #       Test out the greedy recursive algorithm for the binary decision tree
        # Arguments:
        #       None

        # Create a decision tree
        decision_tree = self.binary_decision_trees.greedy_recursive(self.train_data, self.features, self.target,
                                                                    current_depth=0, max_depth=6)

        # Get the classification result of the first row
        classification = self.predict_output.classification_binary_tree(decision_tree, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.accuracy.error_classification_binary_tree(decision_tree, self.test_data)

        # Assert that the classification should be 0.3837785437311504
        self.assertEqual(round(accuracy, 5), round(0.3837785437311504, 5))


    def test_02_greedy_recursive_early_stop(self):
        # Usage:
        #       Test out the greedy recursive algorithm with early stopping
        # Arguments:
        #       None

        # Create a model with max_depth=6, min_node_size=100, min_error_reduction=0
        model_1 = self.binary_decision_trees.greedy_recursive_early_stop(self.train_data, self.features, self.target,
                                                                         max_depth=6, min_node_size=100,
                                                                         min_error_reduction=0.0)

        # Get the classification result of the first row
        classification = self.predict_output.classification_binary_tree(model_1, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.accuracy.error_classification_binary_tree(model_1, self.test_data)

        # Assert that the classification should be 0.38367083153813014
        self.assertEqual(round(accuracy, 5), round(0.38367083153813014, 5))

        # Create a model with max_depth=6, min_node_size=0, min_error_reduction=-1
        model_2 = self.binary_decision_trees.greedy_recursive_early_stop(self.train_data, self.features, self.target,
                                                                         max_depth=6, min_node_size=0,
                                                                         min_error_reduction=-1)

        # Get the classification result of the first row
        classification = self.predict_output.classification_binary_tree(model_2, self.test_data.iloc[0])

        # Assert that the classification should be -1
        self.assertEqual(classification, -1)

        # Compute the accuracy of the decision tree
        accuracy = self.accuracy.error_classification_binary_tree(model_2, self.test_data)

        # Assert that the classification should be 0.3837785437311504
        self.assertEqual(round(accuracy, 5), round(0.3837785437311504, 5))