import numpy as np
from performance_assessment.predict_output import PredictOutput


class Error:
    # Usage:
    #   Used for computing error

    def __init__(self):
        # Usage:
        #       Constructor for the error class, which is mainly used to set predict output class
        # Arguments:
        #       None

        # Create a predict output class
        self.predict_output = PredictOutput()

    def binary_tree(self, tree, data, target):
        # usage:
        #       Computes classification error for binary tree classification, which is based on
        #                           classification error =    # mistakes
        #                                                  ----------------
        #                                                  # total examples
        # Arguments:
        #       tree                 (dict)         : a tree that uses a dictionary format, with is_leaf, prediction,
        #                                             left and right
        #       data                 (pandas frame) : a pandas frame that has the same features binary tree
        #       target               (str)          : the target we want to predict
        # Return:
        #       classification error (float)        : the classification error of the tree

        # Apply the classify(tree, x) to each row in your data
        prediction = data.apply(lambda x: self.predict_output.binary_tree(tree, x), axis=1)

        # Once you've made the predictions, calculate the classification error and return it
        data["prediction"] = prediction
        mistakes = data.apply(lambda x: x[target] != x["prediction"], axis=1).sum()

        # Return mistakes/total examples
        return float(mistakes)/float(len(data))

