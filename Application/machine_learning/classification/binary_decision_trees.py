class BinaryDecisionTrees:
    # Usage:
    #   Greedy Decision Tree algorithm is implemented as:
    #       1. Start with an empty tree
    #       2. Select a feature to split data <- Split according to lowest classification error
    #       For each split:
    #           A. If nothing more to split, make prediction
    #           B. Otherwise, go to step 2 and continue

    def greedy_recursive(self, data, features, target, current_depth=0, max_depth=10):
        # Usage:
        #       Uses recursion to create a tree, create leaf if we encounter:
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

        pass