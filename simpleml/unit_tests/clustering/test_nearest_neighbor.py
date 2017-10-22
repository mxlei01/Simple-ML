"""Implements NearestNeighbor Unittest."""

import unittest
import json
import pandas as pd
from data_extraction import text_analytics
from machine_learning.clustering import nearest_neighbor


class TestNearestNeighbor(unittest.TestCase):

    """Tests for TestNearestNeighbor.

    Uses Wikipedia data to test NearestNeighbor.

    Statics:
        _multiprocess_can_split_ (bool): Flag for nose tests to run tests in parallel.

    """

    _multiprocess_can_split_ = True

    def setUp(self):
        """Set up for TestNearestNeighbor.

        Loads Wikipedia data, and creates training and testing data.

        """
        self.text_analytics = text_analytics.TextAnalytics()
        self.nearest_neighbor = nearest_neighbor.NearestNeighbor()

        # Load the wiki
        self.wiki = pd.read_csv('./unit_tests/test_data/clustering/wiki/people_wiki.csv.bz2')
        self.tf_idf = pd.read_csv('./unit_tests/test_data/clustering/wiki/people_wiki_tf_idf.csv.bz2')

        # Create our word count column on the wiki
        self.wiki["word_count"] = self.wiki["text"].apply(self.text_analytics.word_count)

        # Create our tf idf column on the wiki
        # self.wiki["tf_idf"] = self.wiki["text"].apply(lambda x: self.text_analytics.tf_idf(self.wiki["text"], x))
        self.wiki["tf_idf"] = self.tf_idf["tf_idf"]
        self.wiki["tf_idf"] = self.wiki["tf_idf"].apply(json.loads)

    def test_01_euclidean_distance(self):
        """Test euclidean distance with word count.

        Test euclidean distance with word count and compare to known values.

        """
        # Reduce the frame
        frame = self.wiki[(self.wiki["name"] == "Barack Obama") | (self.wiki["name"] == "George W. Bush")]

        # Get a pandas frame of nearest neighbors
        neighbors = self.nearest_neighbor.nearest_neighbors(frame, "name", "word_count", "euclidean")

        # Get the distance between Barack Obama and George W. Bush
        distance = neighbors[(neighbors["query_label"] == "Barack Obama")
                             & (neighbors["reference_label"] == "George W. Bush")].iloc[0]["distance"]

        # Assert that these two numbers are equal
        self.assertEqual(round(distance, 5), round(34.3947670438, 5))

    def test_02_tf_idf(self):
        """Test tf idf.

        Test tf_idf and compare it to known values.

        """
        # Create corpus
        corpus = pd.Series(["the quick brown fox jumps over lazy dog", "a quick brown dog outpaces a quick fox"])

        # The document is one of the document in a corpus
        document = corpus[0]

        # Compute tf idf value
        tf_idf = self.text_analytics.tf_idf(corpus, document)

        # Assert with known values
        self.assertEqual(round(tf_idf["lazy"], 5), round(0.6931471805599453, 5))

    def test_03_cosine_similarity(self):
        """Test cosine similarity.

        Tests cosine similarity and compare it to known values.

        """
        # Reduce the frame
        frame = self.wiki[(self.wiki["name"] == "Barack Obama") | (self.wiki["name"] == "Joe Biden")]

        # Get a pandas frame of nearest neighbors
        neighbors = self.nearest_neighbor.nearest_neighbors(frame, "name", "tf_idf", "cosine_similarity")

        # Get the distance between Barack Obama and George W. Bush
        distance = neighbors[(neighbors["query_label"] == "Barack Obama")
                             & (neighbors["reference_label"] == "Joe Biden")].iloc[0]["distance"]

        # Assert that these two numbers are equal
        self.assertEqual(round(distance, 5), round(0.703138676734, 5))
