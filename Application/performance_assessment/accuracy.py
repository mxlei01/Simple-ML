"""Implements Accuracy."""

import pandas as pd
from performance_assessment.predict_output import PredictOutput


class Accuracy:

    """Class for computing accuracy.

    Computes accuracy for general method, decision trees, and logistic regression.

    Attributes:
        predict_output (PredictOutput): Class used to predict output.

    """

    def __init__(self):
        """Constructor for Accuracy class to setup predict output.

        Constructor for Accuracy class to setup predict output class.

        """
        self.predict_output = PredictOutput()

    @staticmethod
    def general(predictions, label):
        """Computes general form of accuracy for classification.

        Needs to have predictions and labels before using this function.

        General form of computing accuracy for classification = # Correctly classified data points
                                                                ----------------------------------
                                                                      # Total data points

        Args:
            predictions (numpy.Series): A numpy matrix containing features.
            label (numpy.array): A numpy array containing labels.

        Returns:
            float: Accuracy.

        """
        # Sum the number of correct predictions
        num_correct = (pd.Series(predictions) == pd.Series(label)).sum()

        # Compute the accuracy, which is the number of correct predictions divided by the length of label or prediction
        return num_correct / len(label)

    def logistic_regression(self, feature_matrix, label, coefficients):
        """Computes accuracy for logistic regression.

        Can take in feature matrix and coefficients from logistic regression, and compute accuracy.

        Computes accuracy for classification, which is based on accuracy = # Correctly classified data points
                                                                          ----------------------------------
                                                                                 # Total data points

        Args:
            feature_matrix (numpy.matrix): A numpy matrix containing features.
            label (numpy.array): A numpy array containing labels.
            coefficients (numpy.array): A numpy array containing coefficients.

        Returns:
            float: Accuracy.

        """
        # Get the predictions
        predictions = self.predict_output.logistic_regression(feature_matrix, coefficients)

        # Sum the number of correct predictions
        num_correct = (predictions == label).sum()

        # Compute the accuracy, which is the number of correct predictions divided by the length of feature matrix
        return num_correct / len(feature_matrix)

    @staticmethod
    def decision_tree(data, predictions, target):
        """Computes accuracy for logistic regression.

        Can take in data and predictions along with target from a decision tree to compute accuracy.

        Computes accuracy for decision trees, which is based on accuracy = # Correctly classified data points
                                                                           ----------------------------------
                                                                                  # Total data points

        Args:
            data (pandas.DataFrame): Train/testing data.
            predictions (pandas.Series): A pandas series containing output prediction for data.
            target (str): The target string.

        Returns:
            float: Accuracy.

        """
        # Add the predictions to the data
        data["prediction"] = predictions

        # Calculate the number of mistakes
        mistakes = data.apply(lambda x: x[target] != x["prediction"], axis=1).sum()

        # One minus the mistakes divided by the length of the data
        return 1-(float(mistakes) / float(len(data)))
