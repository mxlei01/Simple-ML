import pandas as pd
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

    def general(self, predictions, label):
        # Usage:
        #       General form of computing accuracy for classification = # correctly classified data points
        #                                                               ----------------------------------
        #                                                                      # total data points
        # Arguments:
        #       prediction (numpy matrix) : a numpy matrix containing features
        #       label      (numpy array)  : a numpy array containing labels
        # Return:
        #       accuracy       (float)

        # Sum the number of correct predictions
        num_correct = (pd.Series(predictions) == pd.Series(label)).sum()

        # Compute the accuracy, which is the number of correct predictions divided by the length of label or prediction
        return num_correct / len(label)

    def logistic_regression(self, feature_matrix, label, coefficients):
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
        predictions = self.predict_output.logistic_regression(feature_matrix, coefficients)

        # Sum the number of correct predictions
        num_correct = (predictions == label).sum()

        # Compute the accuracy, which is the number of correct predictions divided by the length of feature matrix
        return num_correct / len(feature_matrix)

    def decision_tree(self, data, predictions, target):
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
