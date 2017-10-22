"""Implements Nearest Neighbor Clustering."""

import math
from collections import defaultdict
import pandas as pd


class NearestNeighbor:

    """Nearest Neighbor will compute distances from a query point to all other points.

    The Nearest Neighbor is implemented by using a distance metric, such as euclidean or cosine similarity, and
    figures out the nearest neighbors with features such as bag of words or tf-idf.

    """

    def nearest_neighbors(self, data, label, feature, distance):
        """Compute nearest neighbors.

        Compute the nearest neighbors by using the label on data, which tells us the column of the data frame for
        querying, and the feature would be the name of the data frame that uses bag of words or tf-idf, uses the
        distance metric it would compute all of the distance compared to the query point.

        Args:
            data (pandas.DataFrame): Pandas data frame that holds our data.
            label (str): The name of the column that we want to query.
            feature (str): The column name of data that has feature.
            distance (str): Distance metric name.

        Returns:
            neighbors (pandas.DataFrame): A pandas data frame with label, reference, distance, where label is the
                string in the label column, and the reference is the comparison label string, where distance is the
                distance between them.

        """
        neighbors = pd.DataFrame(columns=["query_label", "reference_label", "distance"])

        # Loop through each row in the data frame
        for _, row_target in data.iterrows():
            # Loop through each row in the data frame
            for dummy_, row_compare in data.iterrows():
                # Create a default dictionary and set the query_label as the value in the label column
                comparison = defaultdict()
                comparison["query_label"] = row_target[label]

                # The reference label would be the label value of the row
                comparison["reference_label"] = row_compare[label]

                # Use the distance metric specified by the user
                comparison["distance"] = getattr(self, distance)(row_target[feature], row_compare[feature])

                # Append the new dictionary to the neighbors data frame
                neighbors.loc[len(neighbors.index) + 1] = comparison

        return neighbors

    @staticmethod
    def euclidean(target, compare):
        """Compute the euclidean distance between target and compare.

        The euclidean distance is computed as: √(a_i-a_j)^2+(b_i-b_j)^2

        Args:
            target (dict): Bag of words or td-idf dictionary.
            compare (dict): Bag of words or td-idf dictionary.

        Returns:
            float: Euclidean distance.

        """
        # Convert the default dictionary to a default dictionary
        target = defaultdict(float, target)
        compare = defaultdict(float, compare)

        # Compare the euclidean distance
        return math.sqrt(sum([(target[word] - compare[word]) ** 2 for word in set(target).union(set(compare))]))

    @staticmethod
    def cosine_similarity(target, compare):
        """Compute the cosine similarity between target and compare.

        The cosine similarity is computed as:        x^T*y
                                              1 - ------------
                                                   ||x||*||y|
        Where x^T*y is the multiplication of vectors x and y, essentially Σ x_i*y_i, and ||x|| is the normalization
        of X, which is √(a_i)^2+(b_i)^2.

        Args:
            target (dict): Bag of words or td-idf dictionary.
            compare (dict): Bag of words or td-idf dictionary.

        Returns:
            float: Cosine distance.

        """
        # Convert the default dictionary to a default dictionary
        target = defaultdict(float, target)
        compare = defaultdict(float, compare)

        # Compute x^T*y
        x_y = sum([(target[word] * compare[word]) for word in set(target).union(set(compare))])

        # Compute x norm, ||x||
        x_norm = math.sqrt(sum([target[word] ** 2 for word in target]))

        # Compute y norm, ||y||
        y_norm = math.sqrt(sum([compare[word] ** 2 for word in compare]))

        # Compute cosine distance,       x^T*y
        #                          1 - ------------
        #                               ||x||*||y|
        return 1 - (x_y / (x_norm * y_norm))
