import numpy as np


class PredictOutput:
    # Usage:
    #   Used for predicting outputs based on features and weights

    def regression(self, feature_matrix, weights):
        # Usage:
        #       Predicts output based on y_hat = Hw
        # Arguments:
        #       feature_matrix (numpy matrix) : a numpy matrix containing features
        #       weights        (numpy array)  : a numpy array containing weights
        # Return:
        #       Hw             (numpy array)

        return np.dot(feature_matrix, weights)

    def classification(self, feature_matrix, coefficients):
        # Usage:
        #       Predicts output based on y_i = +1 hw >= 0
        #                                      -1 hw <  0
        # Arguments:
        #       feature_matrix (numpy matrix) : a numpy matrix containing features
        #       label          (numpy array)  : a numpy array containing labels
        #       coefficients   (numpy array)  : a numpy array containing coefficients
        # Return:
        #       Hw             (numpy array)

        return np.dot(feature_matrix, coefficients)

    def classification_binary_tree(self, tree, data_point):
        # Usage:
        #       Classifies a data point (pandas series), by traversing a binary decision tree
        # Arguments:
        #       tree            (dict)          : a tree that uses a dictionary format, with is_leaf, prediction,
        #                                         left and right
        #       data_point      (pandas series) : a pandas series that contains features on the tree
        # Return:
        #       predicted class (string)        : returns the name of the predicted class

        # If the node is the leaf, then we can return the prediction
        if tree['is_leaf']:
            return tree['prediction']
        else:
            # Get the data point according to the splitting feature on the tree
            split_feature_value = data_point[tree['splitting_feature']]

            # If value is equal to 0, then go the left, otherwise right
            if split_feature_value == 0:
                return self.classification_binary_tree(tree['left'], data_point)
            else:
                return self.classification_binary_tree(tree['right'], data_point)
