"""Implements text analytics: word count, and TD-IDF."""

import math
from collections import defaultdict


class TextAnalytics:

    """For analyzes of text data.

    Contains useful functions that can convert text data into numeric data.

    Attributes:
        idf (dict): A dictionary of known idf values for faster lookup.

    """

    def __init__(self):
        """Constructor for TextAnalytics.

        Stores previous idf values for quicker lookup, no need to compute again.

        """
        self.idf = {}

    @staticmethod
    def word_count(text):
        """Computes the word count.

        Computes the word count vector by counting the number of occurrences for a word.

        Args:
            text (str): Text that needs to be converted to numeric vector. Assumes that the input is already cleaned.

        Returns:
            count (dict): A word count dictionary where the key are the word, and the values are the number of
                occurrences for a word.

        """
        # Create a dictionary when access will be initialized as 0 and returning 0 instead of raising an
        # AttributeError
        count = defaultdict(int)

        # Split the text by spaces, and increment the count of the word
        for word in text.split(" "):
            count[word] += 1

        return count

    def tf_idf(self, corpus, document):
        """Computes tf_idf.

        Computes the tf_idf, where tf is the term frequency of each word in a document, and idf is the inverse of
        document frequency, hence          # docs
                                  log -----------------
                                      # docs using word
        We then can compute tf idf using tf*idf, this value is applicable for only one document.

        Args:
            corpus (Pandas.Series): A list of strings (documents).
            document (str): A word document in str format.

        Returns:
            tf_idf (dict) : A dictionary of word and corresponding tf idf value.

        """
        # Compute the word count
        word_count = self.word_count(document)

        # Create a default dictionary, where the default values are float
        idf = defaultdict(float)

        # Split each word in the document
        for word in document.split(" "):
            # Check if the word already exist in our cache, so we don't need to recompute again
            try:
                idf[word] = self.idf[word]
            except KeyError:
                # Corpus is a pandas.Series that has a contains functions that allows regular expression. The \b
                # that wraps around the word would only match word, so it would not match apple with pineapple
                word_appearance = sum(corpus.str.contains(r'\b{}\b'.format(word)))

                # Compute the idf of the word, which is
                #         # docs
                # log -----------------
                #     # docs using word
                idf[word] = math.log(float(len(corpus)) / float(word_appearance))

                # Update our cache
                self.idf[word] = idf[word]

        # Compute the tf idf of the word by multiplying the word_count with idf
        tf_idf = defaultdict(float)
        for word in idf:
            tf_idf[word] = word_count[word] * idf[word]

        return tf_idf
