"""Implements Error."""

from performance_assessment.predict_output import PredictOutput


class Error:

    """Computes error for classification.

    Computes error for classification algorithms, such as binary tree.

    Attributes:
        predict_output (PredictOutput): Class for predicting output.

    """

    def __init__(self):
        """Set up PredictOutput class.

        Constructor for Error, sets up output prediction class.

        """
        self.predict_output = PredictOutput()

    def binary_tree(self, tree, data, target):
        """Compute classification error for binary tree.

        Compute classification error for binary tree classification.
        Classification error =    # Mistakes
                              ----------------
                              # Total examples

        Args:
            tree (dict): The top node of a binary tree, with the following dict format:
                {
                    'is_leaf' (bool): False,
                    'prediction' (NoneType): None,
                    'splitting_feature' (str): splitting_feature,
                    'left' (dict): left_tree,
                    'right' (dict): right_tree
                }
            data (pandas.DataFrame): A pandas frame that has the same features binary tree.
            target (str): The target we want to predict.

        Returns:
            float: Clarification error.

        """
        # Apply the classify(tree, x) to each row in your data
        prediction = data.apply(lambda x: self.predict_output.binary_tree(tree, x), axis=1)

        # Once you've made the predictions, calculate the classification error and return it
        data["prediction"] = prediction
        mistakes = data.apply(lambda x: x[target] != x["prediction"], axis=1).sum()

        # Return mistakes/total examples
        return float(mistakes) / float(len(data))
