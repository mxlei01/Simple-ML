"""Implements BinaryDecisionTrees."""


class BinaryDecisionTrees:

    """A Binary Decision Tree algorithm for building Decision Trees.

    A Binary Decision Tree algorithm are implemented as:
        1. Start with an empty tree.
        2. Select a feature to split data <- Split according to lowest classification error.
            For each split:
                1. If nothing more to split, make prediction.
                2. Otherwise, go to step 2 and continue.

    """

    def greedy_recursive(self, data, features, target, model_parameters):
        """Greedy recursive approach to build a binary decision tree.

        Uses recursion to create a tree, and uses greedy method, which is to get construct nodes that has the lowest
        classification error first.

        Create leaf if we encounter:
           1. No Mistakes after selecting majority class.
           2. No remaining features to split.
           3. Max depth is encountered.
           4. If the left split is equal to the amount of data.
           5. If the right split is equal to the amount of data.
        Otherwise we create a decision stump.

        Args:
            data (pandas.DataFrame): One hot encoded features with target.
            features (list of str): List of features that we will decide to split on.
            target (str): The feature that we want to predict.
            model_parameters (dict): A dictionary of model parameters,
                {
                    current_depth (int): The current depth of the recursion,
                    max_depth (int): The maximum depth that the tree will be created.
                }

        Returns:
            A decision tree root, where an intermediate stump has the following dict:
                {
                    'is_leaf' (bool): False,
                    'prediction' (NoneType): None,
                    'splitting_feature' (str): splitting_feature,
                    'left' (dict): left_tree,
                    'right' (dict): right_tree
                }
            where left_tree and right_tree has the same dictionary format as the
            decision tree root. The splitting_feature tells us the feature
            used to split.

            At the leafs, we use the following dict:
                {
                    'is_leaf' (bool): True
                    'prediction' (int): 1 or -1 depending on prediction,
                    'splitting_feature' (NoneType): None,
                    'left' (NoneType): None,
                    'right' (NoneType): None,
                }

        """
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
        if model_parameters["current_depth"] >= model_parameters["max_depth"]:
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
        left_tree = self.greedy_recursive(left_split, remaining_features, target,
                                          {"current_depth": model_parameters["current_depth"] + 1,
                                           "max_depth": model_parameters["max_depth"]})

        # Create the right tree by doing a recursion
        right_tree = self.greedy_recursive(right_split, remaining_features, target,
                                           {"current_depth": model_parameters["current_depth"] + 1,
                                            "max_depth": model_parameters["max_depth"]})

        # Create a leaf node where this is not a leaf, and with no prediction
        return self.create_node(splitting_feature=splitting_feature, left=left_tree, right=right_tree, is_leaf=False,
                                prediction=None)

    def greedy_recursive_early_stop(self, data, features, target, model_parameters):
        """Greedy recursive approach to build a binary decision tree with early stopping.

        Uses recursion to create a tree, and uses greedy method, which is to get construct nodes that has the lowest
        classification error first. However, we include early stopping features to stop the building of a decision
        tree.

        The early stop includes 3 stopping conditions:
            1. Stop at maximum depth.
            2. Stop if the node has less than minimum node size.
            3. Stop if the error does not reduce.
        Create leaf if we encounter:
           1. No Mistakes after selecting majority class.
           2. No remaining features to split.
           3. Max depth is encountered.
           4. If the left split is equal to the amount of data.
           5. If the right split is equal to the amount of data.
        Otherwise we create a decision stump.

        Args:
            data (pandas.DataFrame): One hot encoded features with target.
            features (list of str): List of features that we will decide to split on.
            target (str): The feature that we want to predict.
            model_parameters (dict): A dictionary of model parameters,
                {
                    current_depth (int): The current depth of the recursion,
                    max_depth (int): The maximum depth that the tree will be created,
                    min_node_size (int): The minimum amount of samples per node,
                    min_error_reduction (float): Minimum error reduction per split.
                }

        Returns:
            A decision tree root, where an intermediate stump has the following dict:
                {
                    'is_leaf' (bool): False,
                    'prediction' (NoneType): None,
                    'splitting_feature' (str): splitting_feature,
                    'left' (dict): left_tree,
                    'right' (dict): right_tree
                }
            where left_tree and right_tree has the same dictionary format as the
            decision tree root. The splitting_feature tells us the feature
            used to split.

            At the leafs, we use the following dict:
                {
                    'is_leaf' (bool): True
                    'prediction' (int): 1 or -1 depending on prediction,
                    'splitting_feature' (NoneType): None,
                    'left' (NoneType): None,
                    'right' (NoneType): None
                }

        """
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
        if model_parameters["current_depth"] >= model_parameters["max_depth"]:
            return self.create_leaf(target_values)

        # 2. Stop if the node has less than minimum node size
        if self.reached_minimum_node_size(data, model_parameters["min_node_size"]):
            return self.create_leaf(target_values)

        # Find the best splitting feature
        splitting_feature = self.best_feature(data, remaining_features, target)

        # Split on the best feature that we found
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]

        # Compute the left and right mistake, and then divide by the amount of data left
        left_mistakes = self.intermediate_node_mistakes(left_split[target])
        right_mistakes = self.intermediate_node_mistakes(right_split[target])
        error_after_split = float((left_mistakes + right_mistakes)) / float(len(data))

        # 3. Stop if the error does not reduce
        # Compute the error before we split, and then divided by the amount of data left
        # If the error before splitting and after splitting is less than specified amount (min_error_reduction)
        # Then we would stop and create a leaf
        if self.error_reduction(float(self.intermediate_node_mistakes(target_values)) / float(len(data)),
                                error_after_split) <= model_parameters["min_error_reduction"]:
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
                                                     {"current_depth": model_parameters["current_depth"] + 1,
                                                      "max_depth": model_parameters["max_depth"],
                                                      "min_node_size": model_parameters["min_node_size"],
                                                      "min_error_reduction": model_parameters["min_error_reduction"]})

        # Create the right tree by doing a recursion
        right_tree = self.greedy_recursive_early_stop(right_split, remaining_features, target,
                                                      {"current_depth": model_parameters["current_depth"] + 1,
                                                       "max_depth": model_parameters["max_depth"],
                                                       "min_node_size": model_parameters["min_node_size"],
                                                       "min_error_reduction": model_parameters["min_error_reduction"]})

        # Create a leaf node where this is not a leaf, and with no prediction
        return self.create_node(splitting_feature=splitting_feature, left=left_tree, right=right_tree, is_leaf=False,
                                prediction=None)

    @staticmethod
    def intermediate_node_mistakes(data_labels):
        """Compute and returns number of errors of a majority class.

        Determine which labels are greater, -1 or 1. If 1 is greater, then return the number of -1 as number of
        mistakes, vice versa, since we will use this function to determine the majority class error. If 1 is greater,
        then we set the prediction to 1, and the number of -1's are mistakes.

        Args:
            data_labels (numpy.array): Array of labels (1 or -1).

        Returns:
            int: The number of errors for a majority class.

        """
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
        """Determine the best feature to split.

        Determine the best splitting label, which we will pick the feature that offers the lowest classification error.
        Classification error =   # mistakes
                              ----------------
                              # total examples

        Args:
            data (pandas.DataFrame): Current node pandas frame that contains one hot encoded features.
            features (list of str): List of feature names.
            target (str): The target label that we are trying to predict.

        Returns:
            best_feature (str): The best feature to split on with the lowest classification error.

        """
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
            error = float(left_mistakes + right_mistakes) / num_data

            # If this is the best error we have found so far,
            # then store the feature as best_feature and the error as best_error
            if error < best_error:
                best_error = error
                best_feature = feature

        return best_feature

    def create_leaf(self, data_labels):
        """Create a leaf node for decision tree algorithm.

        Create a leaf node with prediction.

        Args:
            data_labels (numpy.array): Array of labels (1 or -1).

        Returns:
            leaf node (dict): A leaf node that contains all None except is_leaf which is true, and prediction.
                will be the highest data label present, and has the following dict format:
                {
                    'is_leaf' (bool): True
                    'prediction' (int): 1 or -1 depending on prediction,
                    'splitting_feature' (NoneType): None,
                    'left' (NoneType): None,
                    'right' (NoneType): None,
                }

        """
        # Count the number of data points that are +1 and -1
        num_ones = len(data_labels[data_labels == +1])
        num_minus_ones = len(data_labels[data_labels == -1])

        # For the leaf node, set the prediction to be the majority class
        if num_ones > num_minus_ones:
            prediction = 1
        else:
            prediction = -1

        return self.create_node(splitting_feature=None, left=None, right=None, is_leaf=True, prediction=prediction)

    @staticmethod
    def create_node(splitting_feature, left, right, is_leaf, prediction):
        """Create a leaf node with passed parameters.

        Create a leaf nodes from arguments passed in. This can be used for both decision tree stumps and leaves.

        Args:
            splitting_feature (str or NoneType): The feature that is split in this node, can be None if this is a leaf.
            left (pandas.DataFrame or NoneType): The left node pandas frame, can be None if this is a leaf.
            right (pandas.DataFrame or NoneType): The right node pandas frame, can be None if this is a leaf.
            is_leaf (bool): Flag that indicates if this is a leaf, hence true if leaf, and vice versa.
            prediction (int or NoneType): The prediction value (1 or -1), only if there is a leaf, and None if this
                node is not a leaf.

        Returns:
            leaf node (dict): A leaf node that contains all None except is_leaf which is true, and prediction.
                will be the highest data label present, and has the following dict format:
                {
                    'is_leaf' (bool): flag that indicates if this is a leaf, hence true if leaf, and vice versa
                    'prediction' (int or NoneType): the prediction value (1 or -1), only if there is a leaf, and None
                        if this node is not a leaf,
                    'splitting_feature' (str or NoneType): The feature that is split in this node, can be None if
                        this is a leaf,
                    'left' (pandas.DataFrame or NoneType): the left node pandas frame, can be None if this is a leaf,
                    'right '(pandas.DataFrame or NoneType): the right node pandas frame, can be None if this is a leaf
                }

        """
        return {'splitting_feature': splitting_feature,
                'left': left,
                'right': right,
                'is_leaf': is_leaf,
                'prediction': prediction}

    @staticmethod
    def reached_minimum_node_size(data, min_node_size):
        """Decide if we reached minimum node size.

        Compute whether the number of data points left is less than the minimum node size. Returns true if the
        number of data points is less than or equal to the minimum node size, otherwise false.

        Args:
            data (pandas.DataFrame): Pandas frame that contains one hot encoded features.
            min_node_size (int): Minimum node size for a given node.

        Returns:
            bool: True or false depending on whether the number of data points is less than min_node_size.

        """
        return True if len(data) <= min_node_size else False

    @staticmethod
    def error_reduction(error_before_split, error_after_split):
        """Computes error reduction.

        Computes the error reduction, which is error before the split minus the error after the split.

        Args:
            error_before_split (float): Error before split.
            error_after_split (float): Error after split.

        Returns:
            float: Error reduced if split.

        """
        return error_before_split - error_after_split
