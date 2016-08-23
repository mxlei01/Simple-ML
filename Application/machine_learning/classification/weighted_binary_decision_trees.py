class WeightedBinaryDecisionTrees:
    # Usage:
    #       A Weighted Binary Decision Tree algorithm is implemented as:
    #           1. Start with an empty tree
    #           2. Select a feature to split data <- Split according to lowest weighted error
    #           For each split:
    #               A. If nothing more to split, make prediction
    #               B. Otherwise, go to step 2 and continue

    def fit(self, data, features, target, data_weights, current_depth=0, max_depth=10, minimum_error=1e-15):
        # Usage:
        #       Uses recursion to create a tree, and uses greedy method, which is to get construct
        #       nodes that has the lowest weighted_error error first.
        #       Create leaf if we encounter:
        #           1. No weighted error after selecting majority class
        #           2. No remaining features to split
        #           3. Max depth is encountered
        #           4. If the left split is equal to the amount of data
        #           5. If the right split is equal to the amount of data
        # Arguments:
        #       data               (pandas frame)  : one hot encoded features with target
        #       features           (list)          : list of features that we will decide to split on
        #       target             (string)        : the feature that we want to predict
        #       data_weights       (pandas series) : pandas series of weights for corresponding label
        #       current_depth      (int)           : the current depth of the recursion
        #       max_depth          (int)           : the maximum depth that the tree will be created
        #       minimum_error      (float)         : the minimum error to count as no error
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

        # 1. No weighted error after selecting majority class
        if self.intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= minimum_error:
            return self.create_leaf(target_values, data_weights)

        # 2. No remaining features to split
        if len(remaining_features) == 0:
            return self.create_leaf(target_values, data_weights)

        # 3. Max depth is encountered
        if current_depth >= max_depth:
            return self.create_leaf(target_values, data_weights)

        # Find the best splitting feature
        splitting_feature = self.best_feature(data, remaining_features, target, data_weights)
        remaining_features.remove(splitting_feature)

        # Split on the best feature that we found
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]

        # Split on the weights
        left_data_weights = data_weights[data[splitting_feature] == 0]
        right_data_weights = data_weights[data[splitting_feature] == 1]

        # 4. If the left split is equal to the amount of data
        # This is done since if the feature split has exactly the same data, then there's nothing we need to do
        if len(left_split) == len(data):
            return self.create_leaf(left_split[target], data_weights)

        # 5. If the right split is equal to the amount of data
        if len(right_split) == len(data):
            return self.create_leaf(right_split[target], data_weights)

        # Create the left tree by doing a recursion
        left_tree = self.fit(left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
        # Create the right tree by doing a recursion
        right_tree = self.fit(right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)

        # Create a leaf node where this is not a leaf, and with no prediction
        return self.leaf_node(splitting_feature=splitting_feature, left=left_tree, right=right_tree, is_leaf=False,
                              prediction=None)

    def intermediate_node_weighted_mistakes(self, data_labels, data_weights):
        # Usage:
        #       Determine which labels (+1 or -1) have the lowest weighted errors. We will return 1 is +1 has the
        #       lowest weighted error.
        # Arguments:
        #       data_labels          (numpy array)     : array of labels (1 or -1)
        #       data_weights         (pandas series)   : pandas series of weights for corresponding label
        # Return:
        #       weight, label        (weight, label)   : the weight and label

        # Sum of the weight where the label data are == 1, which means the weight of mistakes if we chose -1
        weighted_mistakes_negative = sum(data_weights[data_labels == 1])

        # Sum of the weight where the label data are == -1, which means the weight of mistakes if we chose +1
        weighted_mistakes_positive = sum(data_weights[data_labels != 1])

        # Return the tuple of (weight, label), where we return (weight, -1) if weighted_mistakes_negative was less
        # than +1 label, and vice versa.
        return (weighted_mistakes_negative, -1) if weighted_mistakes_negative < weighted_mistakes_positive \
            else (weighted_mistakes_positive, 1)

    def best_feature(self, data, features, target, data_weights):
        # Usage:
        #       Determines the best splitting label, which we will pick the feature that offers the lowest
        #       Weighted error = sum weighted error
        #                        ------------------
        #                         # total examples
        # Arguments:
        #       data         (pandas frame)  : current node pandas frame that contains one hot encoded features
        #       features     (list)          : list of feature names
        #       target       (string)        : the target label that we are trying to predict
        #       data_weights (pandas series) : pandas series of weights for corresponding label
        # Returns:
        #       best_feature (string)       : the best feature to split on with the lowest weighted error

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

            # Apply the same filtering to data_weights to create left_data_weights, right_data_weights
            left_data_weights = data_weights[data[feature] == 0]
            right_data_weights = data_weights[data[feature] == 1]

            # Calculate the weight of mistakes for left and right sides
            left_weighted_mistakes, left_class = self.intermediate_node_weighted_mistakes(left_split[target],
                                                                                          left_data_weights)
            right_weighted_mistakes, right_class = self.intermediate_node_weighted_mistakes(right_split[target],
                                                                                            right_data_weights)

            # Compute the weighted error of this split.
            # error = # of left mistakes + # of right mistakes
            #         ----------------------------------------
            #                     total examples
            error = float(left_weighted_mistakes + right_weighted_mistakes)/num_data

            # If this is the best error we have found so far,
            # then store the feature as best_feature and the error as best_error
            if error < best_error:
                best_error = error
                best_feature = feature

        # Return the best feature we found
        return best_feature

    def create_leaf(self, data_labels, data_weights):
        # Usage:
        #       Creates a leaf node with prediction
        # Arguments:
        #       data_labels (numpy array) : array of labels (1 or -1)
        #       data_weights (pandas series) : pandas series of weights for corresponding label
        # Returns:
        #       leaf node   (dict)        : a leaf node that contains all None except is_leaf which is true, and
        #                                   prediction will be the lowest weighted error for a split on +1 or -1

        # Computed weight of mistakes.
        weighted_error, best_class = self.intermediate_node_weighted_mistakes(data_labels, data_weights)

        # Return the leaf node
        return self.leaf_node(splitting_feature=None, left=None, right=None, is_leaf=True, prediction=best_class)

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
