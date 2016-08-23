class BinaryDecisionTrees:
    # Usage:
    #       A Binary Decision Tree algorithm is implemented as:
    #           1. Start with an empty tree
    #           2. Select a feature to split data <- Split according to lowest classification error
    #           For each split:
    #               A. If nothing more to split, make prediction
    #               B. Otherwise, go to step 2 and continue

    def greedy_recursive(self, data, features, target, current_depth=0, max_depth=10):
        # Usage:
        #       Uses recursion to create a tree, and uses greedy method, which is to get construct
        #       nodes that has the lowest classification error first.
        #       Create leaf if we encounter:
        #           1. No Mistakes after selecting majority class
        #           2. No remaining features to split
        #           3. Max depth is encountered
        #           4. If the left split is equal to the amount of data
        #           5. If the right split is equal to the amount of data
        # Arguments:
        #       data               (pandas frame) : one hot encoded features with target
        #       features           (list)         : list of features that we will decide to split on
        #       target             (string)       : the feature that we want to predict
        #       current_depth      (int)          : the current depth of the recursion
        #       max_depth          (int)          : the maximum depth that the tree will be created
        # Return:
        #       decision tree root (dict) : a decision tree root, where an intermediate stump has the following dict:
        #                                       {'is_leaf'          : False,
        #                                        'prediction'       : None,
        #                                        'splitting_feature': splitting_feature,
        #                                        'left'             : left_tree,
        #                                        'right'            : right_tree}
        #                                    where left_tree and right_tree has the same dictionary format as the
        #                                    decision tree root. The splitting_feature tells us the feature
        #                                    used to split.
        #                                    At the leafs, we use the following dict:
        #                                       {'splitting_feature' : None,
        #                                        'prediction': 1 or -1,
        #                                        'left' : None,
        #                                        'right' : None,
        #                                        'is_leaf': True}
        #                                    We will have the prediction equal to -1 or 1

        # Make a copy of the features
        remaining_features = features[:]

        # Get the data from the target column. Notice that that every recursion we do, we will get less portion
        # of the target_values, even though target never changes, since the left_split and right_split will
        # be called on the next recursion, which is the subset of the original data (pandas frame with all features
        # one hot encoded)
        target_values = data[target]

        # 1. No Mistakes after selecting majority class
        if self.intermediate_node_mistakes(target_values) == 0:
            return self.create_leaf(target_values)

        # 2. No remaining features to split
        if len(remaining_features) == 0:
            return self.create_leaf(target_values)

        # 3. Max depth is encountered
        if current_depth >= max_depth:
            return self.create_leaf(target_values)

        # Find the best splitting feature
        splitting_feature = self.best_feature(data, remaining_features, target)

        # Split on the best feature that we found
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
        remaining_features.remove(splitting_feature)

        # 4. If the left split is equal to the amount of data
        # This is done since if the feature split has exactly the same data, then there's nothing we need to do
        if len(left_split) == len(data):
            return self.create_leaf(left_split[target])

        # 5. If the right split is equal to the amount of data
        if len(right_split) == len(data):
            return self.create_leaf(right_split[target])

        # Create the left tree by doing a recursion
        left_tree = self.greedy_recursive(left_split, remaining_features, target, current_depth + 1, max_depth)
        # Create the right tree by doing a recursion
        right_tree = self.greedy_recursive(right_split, remaining_features, target, current_depth + 1, max_depth)

        # Create a leaf node where this is not a leaf, and with no prediction
        return self.leaf_node(splitting_feature=splitting_feature, left=left_tree, right=right_tree, is_leaf=False,
                              prediction=None)

    def greedy_recursive_early_stop(self, data, features, target, current_depth=0, max_depth=10,
                                    min_node_size=1, min_error_reduction=0.0):
        # Usage:
        #       Uses recursion to create a tree, and uses greedy method, which is to get construct
        #       nodes that has the lowest classification error first.
        #       The early stop includes 3 stopping conditions:
        #           1. Stop at maximum depth
        #           2. Stop if the node has less than minimum node size
        #           3. Stop if the error does not reduce
        #       Create leaf if we encounter:
        #           1. No Mistakes after selecting majority class
        #           2. No remaining features to split
        #           3. If the left split is equal to the amount of data
        #           4. If the right split is equal to the amount of data
        # Arguments:
        #       data               (pandas frame) : one hot encoded features with target
        #       features           (list)         : list of features that we will decide to split on
        #       target             (string)       : the feature that we want to predict
        #       current_depth      (int)          : the current depth of the recursion
        #       max_depth          (int)          : the maximum depth that the tree will be created
        # Return:
        #       decision tree root (dict) : a decision tree root, where an intermediate stump has the following dict:
        #                                       {'is_leaf'          : False,
        #                                        'prediction'       : None,
        #                                        'splitting_feature': splitting_feature,
        #                                        'left'             : left_tree,
        #                                        'right'            : right_tree}
        #                                    where left_tree and right_tree has the same dictionary format as the
        #                                    decision tree root. The splitting_feature tells us the feature
        #                                    used to split.
        #                                    At the leafs, we use the following dict:
        #                                       {'splitting_feature' : None,
        #                                        'prediction': 1 or -1,
        #                                        'left' : None,
        #                                        'right' : None,
        #                                        'is_leaf': True}
        #                                    We will have the prediction equal to -1 or 1

        # Make a copy of the features
        remaining_features = features[:]

        # Get the data from the target column. Notice that that every recursion we do, we will get less portion
        # of the target_values, even though target never changes, since the left_split and right_split will
        # be called on the next recursion, which is the subset of the original data (pandas frame with all features
        # one hot encoded)
        target_values = data[target]

        # 1. No Mistakes after selecting majority class
        if self.intermediate_node_mistakes(target_values) == 0:
            return self.create_leaf(target_values)

        # 2. No remaining features to split
        if len(remaining_features) == 0:
            return self.create_leaf(target_values)

        # 1. Stop at maximum depth
        if current_depth >= max_depth:
            return self.create_leaf(target_values)

        # 2. Stop if the node has less than minimum node size
        if self.reached_minimum_node_size(data, min_node_size):
            return self.create_leaf(target_values)

        # Find the best splitting feature
        splitting_feature = self.best_feature(data, remaining_features, target)

        # Split on the best feature that we found
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]

        # 3. Stop if the error does not reduce
        # Compute the error before we split, and then divided by the amount of data left
        error_before_split = float(self.intermediate_node_mistakes(target_values))/float(len(data))

        # Compute the left and right mistake, and then divide by the amount of data left
        left_mistakes = self.intermediate_node_mistakes(left_split[target])
        right_mistakes = self.intermediate_node_mistakes(right_split[target])
        error_after_split = float((left_mistakes + right_mistakes)) / float(len(data))

        # If the error before splitting and after splitting is less than specified amount (min_error_reduction)
        # Then we would stop and create a leaf
        if self.error_reduction(error_before_split, error_after_split) <= min_error_reduction:
            return self.create_leaf(target_values)

        # Remove the splitting feature from the remaining features
        remaining_features.remove(splitting_feature)

        # 4. If the left split is equal to the amount of data
        # This is done since if the feature split has exactly the same data, then there's nothing we need to do
        if len(left_split) == len(data):
            return self.create_leaf(left_split[target])

        # 5. If the right split is equal to the amount of data
        if len(right_split) == len(data):
            return self.create_leaf(right_split[target])

        # Create the left tree by doing a recursion
        left_tree = self.greedy_recursive_early_stop(left_split, remaining_features, target,
                                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)
        # Create the right tree by doing a recursion
        right_tree = self.greedy_recursive_early_stop(right_split, remaining_features, target,
                                                      current_depth + 1, max_depth, min_node_size, min_error_reduction)

        # Create a leaf node where this is not a leaf, and with no prediction
        return self.leaf_node(splitting_feature=splitting_feature, left=left_tree, right=right_tree, is_leaf=False,
                              prediction=None)

    def intermediate_node_mistakes(self, data_labels):
        # Usage:
        #       Determine which labels are greater, -1 or 1. If 1 is greater, then return the number of
        #       -1 as number of mistakes, vice versa, since we will use this function to determine the majority class
        #       error. If 1 is greater, then we set the prediction to 1, and the number of -1's are mistakes.
        # Arguments:
        #       data_labels          (numpy array) : array of labels (1 or -1)
        # Return:
        #       majority class error (int)         : the number of errors

        # Corner case: If labels_in_node is empty, return 0
        if len(data_labels) == 0:
            return 0

        # Count the number of 1's
        total_one = (data_labels == 1).sum()

        # Count the number of -1's
        total_negative_one = (data_labels == -1).sum()

        # Return the number of mistakes that the majority classifier makes
        return total_negative_one if total_one > total_negative_one else total_one

    def best_feature(self, data, features, target):
        # Usage:
        #       Determines the best splitting label, which we will pick the feature that offers the lowest
        #       classification error = # mistakes
        #                              ----------------
        #                              # total examples
        # Arguments:
        #       data         (pandas frame) : current node pandas frame that contains one hot encoded features
        #       features     (list)         : list of feature names
        #       target       (string)       : the target label that we are trying to predict
        # Returns:
        #       best_feature (string)       : the best feature to split on with the lowest classification error

        # Keep track of best feature and lowest error, since error is always less than 1, we need to setup
        # best_error that is greater than 1
        best_feature = None
        best_error = 2.0

        # Convert to float to make sure error gets computed correctly.
        num_data = float(len(data))

        # Loop through each feature to consider splitting on that feature
        for feature in features:

            # The left split will have all data points where the feature value is 0
            left_split = data[data[feature] == 0]

            # The right split will have all data points where the feature value is 1
            right_split = data[data[feature] == 1]

            # Calculate the number of misclassified examples in the left split
            left_mistakes = self.intermediate_node_mistakes(left_split[target])

            # Calculate the number of misclassified examples in the right split
            right_mistakes = self.intermediate_node_mistakes(right_split[target])

            # Compute the classification error of this split.
            # error = # of left mistakes + # of right mistakes
            #         ----------------------------------------
            #                     total examples
            error = float(left_mistakes + right_mistakes)/num_data

            # If this is the best error we have found so far,
            # then store the feature as best_feature and the error as best_error
            if error < best_error:
                best_error = error
                best_feature = feature

        # Return the best feature we found
        return best_feature

    def create_leaf(self, data_labels):
        # Usage:
        #       Creates a leaf node with prediction
        # Arguments:
        #       data_labels (numpy array) : array of labels (1 or -1)
        # Returns:
        #       leaf node   (dict)        : a leaf node that contains all None except is_leaf which is true, and
        #                                   prediction will be the highest data label present

        # Count the number of data points that are +1 and -1
        num_ones = len(data_labels[data_labels == +1])
        num_minus_ones = len(data_labels[data_labels == -1])

        # For the leaf node, set the prediction to be the majority class
        prediction = None
        if num_ones > num_minus_ones:
            prediction = 1
        else:
            prediction = -1

        # Return the leaf node
        return self.leaf_node(splitting_feature=None, left=None, right=None, is_leaf=True, prediction=prediction)

    def leaf_node(self, splitting_feature, left, right, is_leaf, prediction):
        # Usage:
        #       Creates a leaf node
        # Arguments:
        #       splitting_feature (string)       : the feature that is split in this node
        #       left              (pandas frame) : the left node pandas frame
        #       right             (pandas frame) : the right node pandas frame
        #       is_leaf           (boolean)      : flag that indicates if this is a leaf
        #       prediction        (int)          : the prediction value (1 or -1), only if there is a leaf
        # Return:
        #       leaf node         (dict)         : a dictionary using the input arguments

        # Create a dictionary out of the input arguments, and return it
        return {'splitting_feature': splitting_feature,
                'left': left,
                'right': right,
                'is_leaf': is_leaf,
                'prediction': prediction}

    def reached_minimum_node_size(self, data, min_node_size):
        # Usage:
        #       Computes whether the number of data points left is less than the minimum nodes size
        # Arguments:
        #       data          (pandas frame) : pandas frame that contains one hot encoded features
        #       min_node_size (int)          : minimum node size for a given node
        # Returns:
        #       true or false (boolean)      : depending on whether the number of data points is less than
        #                                      min_node_size

        # Return True if the number of data points is less than or equal to the minimum node size
        return True if len(data) <= min_node_size else False

    def error_reduction(self, error_before_split, error_after_split):
        # Usage:
        #       Computes the error reduction after and before the split
        # Arguments:
        #       error_before_split (float) : error before split
        #       error_after_split  (float) : error after split
        # Returns:
        #       error reduced      (float) : error reduced if split

        # Return the error before the split minus the error after the split
        return error_before_split-error_after_split
