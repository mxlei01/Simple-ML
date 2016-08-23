import numpy as np
from performance_assessment.predict_output import PredictOutput


class Accuracy:
    # Usage:
    #   Used for computing accuracy

    def __init__(self):
        # Usage:
        #       Constructor for the accuracy class, which is mainly used to set predict output class
        # Arguments:
        #       None

        # Create a predict output class
        self.predict_output = PredictOutput()

    def accuracy_classification(self, feature_matrix, label, coefficients):
        # Usage:
        #       Computes accuracy for classification, which is based on accuracy = # correctly classified data points
        #                                                                          ----------------------------------
        #                                                                                 # total data points
        # Arguments:
        #       feature_matrix (numpy matrix) : a numpy matrix containing features
        #       label          (numpy array)  : a numpy array containing labels
        #       coefficients   (numpy array)  : a numpy array containing coefficients
        # Return:
        #       accuracy       (float)

        # Get the predictions
        predictions = self.predict_output.classification(feature_matrix, coefficients)

        # The apply_threshold will create an array of 1 or -1 depending on the predictions
        apply_threshold = np.vectorize(lambda x: 1. if x > 0 else -1.)

        # Apply the function to the predictions
        predictions = apply_threshold(predictions)

        # Sum the number of correct predictions
        num_correct = (predictions == label).sum()

        # Compute the accuracy, which is the number of correct predictions divided by the length of feature matrix
        return num_correct / len(feature_matrix)

    def accuracy_classification_decision_tree(self, data, predictions, target):
        # Usage:
        #       Computes accuracy for decision trees, which is based on accuracy = # correctly classified data points
        #                                                                          ----------------------------------
        #                                                                                 # total data points
        # Arguments:
        #       data        (pandas frame)  : train/testing data
        #       predictions (pandas series) : a pandas series containing output prediction for data
        #       target      (string)        : the target string
        # Return:
        #       accuracy    (float)

        # Add the predictions to the data
        data["prediction"] = predictions

        # Calculate the number of mistakes
        mistakes = data.apply(lambda x: x[target] != x["prediction"], axis=1).sum()

        # One minus the mistakes divided by the length of the data
        return 1-(float(mistakes)/float(len(data)))

    def error_classification_binary_tree(self, tree, data, target):
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
        prediction = data.apply(lambda x: self.predict_output.classification_binary_tree(tree, x), axis=1)

        # Once you've made the predictions, calculate the classification error and return it
        data["prediction"] = prediction
        mistakes = data.apply(lambda x: x[target] != x["prediction"], axis=1).sum()

        # Return mistakes/total examples
        return float(mistakes)/float(len(data))

