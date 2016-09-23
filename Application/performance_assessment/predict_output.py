import numpy as np
import pandas as pd


class PredictOutput:
    """Class for predicting outputs.

    Used for predicting outputs based on features and weights, supports different types of algorithms such as
    regression, logistic regression and adaboost.

    """

    @staticmethod
    def regression(feature_matrix, weights):
        """Predicts output for regression.

        Predicts output based on y_hat = Hw

        Args:
            feature_matrix (numpy.matrix): A numpy matrix containing features.
            weights (numpy.array): A numpy array containing weights.

        Returns:
            numpy.array: Hw

        """
        return np.dot(feature_matrix, weights)

    @staticmethod
    def logistic_regression(feature_matrix, coefficients, threshold=0):
        """Predicts output for logistic regression.

        Predicts output based on y_i = +1 hw >= 0
                                       -1 hw <  0

        Args:
            feature_matrix (numpy matrix): A numpy matrix containing features.
            coefficients (numpy array): A numpy array containing coefficients.
            threshold (int): A threshold to determine 1, or -1.

        Returns:
            numpy.array: T(Hw), The feature matrix dot product with coefficients and then applied threshold if the
                value is greater than 0, then return 1, else -1

        """
        # The apply_threshold will create an array of 1 or -1 depending on the predictions
        apply_threshold = np.vectorize(lambda x: 1. if x > threshold else -1.)

        # Apply the apply_threshold to the predictions
        return apply_threshold(np.dot(feature_matrix, coefficients))

    def binary_tree(self, tree, data_point):
        """Predicts output for binary tree.

        Classifies a data point (pandas series), by traversing a binary decision tree

        Args:
            tree (dict): The top node of a binary tree, with the following dict format:
                {
                    'is_leaf' (bool): False,
                    'prediction' (NoneType): None,
                    'splitting_feature' (str): splitting_feature,
                    'left' (dict): left_tree,
                    'right' (dict): right_tree
                }
            data_point (pandas.Series): A pandas series that contains features on the tree.

        Returns:
            str: name of the predicted class

        """
        # If the node is the leaf, then we can return the prediction
        if tree['is_leaf']:
            return tree['prediction']
        else:
            # Get the data point according to the splitting feature on the tree
            split_feature_value = data_point[tree['splitting_feature']]

            # If value is equal to 0, then go the left, otherwise right
            if split_feature_value == 0:
                return self.binary_tree(tree['left'], data_point)
            else:
                return self.binary_tree(tree['right'], data_point)

    @staticmethod
    def adaboost_binary_decision_tree(prediction_method, models, weights, data):
        """Predicts output for adaboost with binary decision tree.

        Classifies a data point for adaboost algorithm by computing the sign of the weighted result.

        Args:
            prediction_method (func): Function to predict output data.
            models (List of obj): List of models computed by adaboost.
            weights (list of numpy.array): List of weights computed by adaboost.
            data (pandas.DataFrame) : A pandas frame that contains training/testing data.

        Returns:
            scores (pandas.Series): Outputs for each set of feature.

        """
        # Create scores equal to the length of data
        scores = pd.Series([0.] * len(data))

        # Loop through each models
        for i, model in enumerate(models):
            predictions = data.apply(lambda x, m=model: prediction_method(m, x), axis=1)

            # Accumulate predictions on scores array
            predictions = predictions.apply(lambda x, w=weights: x * w[i])
            scores = scores + predictions

        # Return the prediction of each data
        return scores.apply(lambda score: +1 if score > 0 else -1)

    @staticmethod
    def adaboost_logistic_regression(prediction_method, models, weights, feature_matrix):
        """Predicts output for adaboost with logistic regression.

        Classifies a data point for adaboost algorithm by computing the sign of the weighted result.

        Args:
            prediction_method (func): Function to predict output data.
            models (list of obj): List of models computed by adaboost.
            weights (list of numpy.array): List of weights computed by adaboost.
            feature_matrix (numpy.ndarray): Features of a dataset.

        Returns:
            scores (pandas.Series): Outputs for each set of feature.

        """
        # Create scores equal to the length of data
        scores = pd.Series([0.] * len(feature_matrix))

        # Loop through each models
        for i, model in enumerate(models):
            predictions = pd.Series(prediction_method(feature_matrix, model))

            # Accumulate predictions on scores array
            predictions = predictions.apply(lambda x, w=weights: x * w[i])
            scores = scores + predictions

        # Return the prediction of each data
        return scores.apply(lambda score: +1 if score > 0 else -1)
