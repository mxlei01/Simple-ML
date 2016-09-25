"""Implements TestLogisticRegression Unittest."""

import json
import string
import unittest
import numpy as np
import pandas as pd
from data_extraction.convert_numpy import ConvertNumpy
from machine_learning.classification.logistic_regression import LogisticRegression
from ml_math.log_likelihood import LogLikelihood
from performance_assessment.predict_output import PredictOutput
from performance_assessment.confusion_matrix import ConfusionMatrix


class TestLogisticRegression(unittest.TestCase):

    """Tests for LogisticRegression class.

    Uses Amazon data to test logistic regression.

    """

    def setUp(self):
        """Constructor for TestLogisticRegression.

        Loads Amazon data, and creates training and testing data.

        """
        self.convert_numpy = ConvertNumpy()
        self.log_likelhood = LogLikelihood()
        self.predict_output = PredictOutput()
        self.logistic_regression = LogisticRegression()
        self.confusion_matrix = ConfusionMatrix()

        # Load the important words
        self.important_words = json.load(open('./unit_tests/test_data/classification/amazon/important_words.json', 'r'))

        # Load the amazon baby subset
        self.review_frame = pd.read_csv('./unit_tests/test_data/classification/amazon/amazon_baby_subset.csv')

        # Review needs to be text
        self.review_frame['review'].astype(str)

        # Clean up the punctuations
        self.review_frame['review_clean'] = self.review_frame.apply(
            axis=1,
            func=lambda row: str(row["review"]).translate(str.maketrans({key: None for key in string.punctuation})))

        # Remove any nan text
        self.review_frame['review_clean'] = self.review_frame.apply(
            axis=1,
            func=lambda row: '' if row["review_clean"] == "nan" else row["review_clean"])

        # Count the number of words that appears in each review, and make an indepedent column
        for word in self.important_words:
            self.review_frame[word] = self.review_frame['review_clean'].apply(lambda s, w=word: s.split().count(w))

        # Load training data
        self.train_frame = pd.read_csv('./unit_tests/test_data/classification/amazon/amazon_baby_subset_train_mod2.csv')

    def test_01_gradient_ascent(self):
        """Test gradient ascent algorithm.

        Tests the gradient ascent algorithm and compare it with known values.

        """
        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix, sentiment = self.convert_numpy.convert_to_numpy(self.review_frame, features, output, 1)

        # Compute the coefficients
        coefficients = self.logistic_regression.gradient_ascent(feature_matrix, sentiment,
                                                                initial_coefficients=np.zeros(194),
                                                                step_size=1e-7, max_iter=301)

        # Real coefficients that we need to compare with the computed coefficients
        real_coef = [5.16220157e-03,   1.55656966e-02,  -8.50204675e-03,
                     6.65460842e-02,   6.58907629e-02,   5.01743882e-03,
                     -5.38601484e-02,  -3.50488413e-03,   6.47945868e-02,
                     4.54356263e-02,   3.98353364e-03,   2.00775410e-02,
                     3.01350011e-02,  -2.87115530e-02,   1.52161964e-02,
                     2.72592062e-04,   1.19448177e-02,  -1.82461935e-02,
                     -1.21706420e-02,  -4.15110334e-02,   2.76820391e-03,
                     1.77031999e-02,  -4.39700067e-03,   4.49764014e-02,
                     9.90916464e-03,   8.99239081e-04,  -1.36219516e-03,
                     1.26859357e-02,   8.26466695e-03,  -2.77426972e-02,
                     6.10128809e-04,   1.54084501e-02,  -1.32134753e-02,
                     -3.00512492e-02,   2.97399371e-02,   1.84087080e-02,
                     2.86178752e-03,  -1.05768015e-02,  -6.57350362e-04,
                     -1.01476555e-02,  -4.79579528e-03,   7.50891810e-03,
                     4.27938289e-03,   3.06785501e-03,  -2.20317661e-03,
                     9.57273354e-03,   9.91666827e-05,  -1.98462567e-02,
                     1.75702722e-02,   1.55478612e-03,  -1.77375440e-02,
                     9.78324102e-03,   1.17031606e-02,  -7.35345937e-03,
                     -6.08714030e-03,   6.43766808e-03,   1.07159665e-02,
                     -3.05345476e-03,   7.17190727e-03,   5.73320003e-03,
                     4.60661876e-03,  -5.20588421e-03,   6.71012331e-03,
                     9.03281814e-03,   1.74563147e-03,   6.00279979e-03,
                     1.20181744e-02,  -1.83594607e-02,  -6.91010811e-03,
                     -1.38687273e-02,  -1.50406590e-02,   5.92353611e-03,
                     5.67478991e-03,  -5.28786220e-03,   3.08147864e-03,
                     5.53751236e-03,   1.49917916e-02,  -3.35666000e-04,
                     -3.30695153e-02,  -4.78990943e-03,  -6.41368859e-03,
                     7.99938935e-03,  -8.61390444e-04,   1.68052959e-02,
                     1.32539901e-02,   1.72307051e-03,   2.98030675e-03,
                     8.58284300e-03,   1.17082481e-02,   2.80825907e-03,
                     2.18724016e-03,   1.68824711e-02,  -4.65973741e-03,
                     1.51368285e-03,  -1.09509122e-02,   9.17842898e-03,
                     -1.88572281e-04,  -3.89820373e-02,  -2.44821005e-02,
                     -1.87023714e-02,  -2.13943485e-02,  -1.29690465e-02,
                     -1.71378670e-02,  -1.37566767e-02,  -1.49770449e-02,
                     -5.10287978e-03,  -2.89789761e-02,  -1.48663194e-02,
                     -1.28088380e-02,  -1.07709355e-02,  -6.95286915e-03,
                     -5.04082164e-03,  -9.25914404e-03,  -2.40427481e-02,
                     -2.65927785e-02,  -1.97320937e-03,  -5.04127508e-03,
                     -7.00791912e-03,  -3.48088523e-03,  -6.40958916e-03,
                     -4.07497010e-03,  -6.30054296e-03,  -1.09187932e-02,
                     -1.26051900e-02,  -1.66895314e-03,  -7.76418781e-03,
                     -5.15960485e-04,  -1.94199551e-03,  -1.24761586e-03,
                     -5.01291731e-03,  -9.12049191e-03,  -7.22098801e-03,
                     -8.31782981e-03,  -5.60573348e-03,  -1.47098335e-02,
                     -9.31520819e-03,  -2.22034402e-03,  -7.07573098e-03,
                     -5.10115608e-03,  -8.93572862e-03,  -1.27545713e-02,
                     -7.04171991e-03,  -9.76219676e-04,   4.12091713e-04,
                     8.29251160e-04,   2.64661064e-03,  -7.73228782e-03,
                     1.53471164e-03,  -7.37263060e-03,  -3.73694386e-03,
                     -3.81416409e-03,  -1.64575145e-03,  -3.31887732e-03,
                     1.22257832e-03,   1.36699286e-05,  -3.01866601e-03,
                     -1.02826343e-02,  -1.06691327e-02,   2.23639046e-03,
                     -9.87424798e-03,  -1.02192048e-02,  -3.41330929e-03,
                     3.34489960e-03,  -3.50984516e-03,  -6.26283150e-03,
                     -7.22419943e-03,  -5.47016154e-03,  -1.25063947e-02,
                     -2.47805699e-03,  -1.60017985e-02,  -6.40098934e-03,
                     -4.26644386e-03,  -1.55376990e-02,   2.31349237e-03,
                     -9.06653337e-03,  -6.30012672e-03,  -1.21010303e-02,
                     -3.02578875e-03,  -6.76289718e-03,  -5.65498722e-03,
                     -6.87050239e-03,  -1.18950595e-02,  -1.86489236e-04,
                     -1.15230476e-02,   2.81533219e-03,  -8.10150295e-03,
                     -1.00062131e-02,   4.02037651e-03,  -5.44300346e-03,
                     2.85818985e-03,   1.19885003e-04,  -6.47587687e-03,
                     -1.14493516e-03,  -7.09205934e-03]

        # Loop through each value, the coefficients must be the same
        for pred_coef, coef in zip(coefficients, real_coef):

            # Assert that both values are the same
            self.assertEqual(round(pred_coef, 5), round(coef, 5))

        # Get the output of the logistic regression with threshold 0
        output = self.predict_output.logistic_regression(feature_matrix, coefficients, 0)

        # Generate a confusion matrix
        confusion_matrix = self.confusion_matrix.confusion_matrix(sentiment, output)

        # Assert the values are to be expected
        self.assertEqual(confusion_matrix, {'false_negatives': 7311, 'true_negatives': 20635,
                                            'true_positives': 19268, 'false_positives': 5858})

        # Assert that the precision is correct
        self.assertEqual(round(self.confusion_matrix.precision(sentiment, output), 5),
                         round(0.7249332179540239, 5))

        # Assert that the recall is correct
        self.assertEqual(round(self.confusion_matrix.recall(sentiment, output), 5),
                         round(0.7668550505452519, 5))

    def test_02_stochastic_gradient_ascent(self):
        """Test stochastic gradient descent for logistic regression.

        Tests stochastic gradient descent and test it with some known values.

        """
        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix, sentiment = self.convert_numpy.convert_to_numpy(self.train_frame, features, output, 1)

        # Compute the coefficients
        coefficients = self.logistic_regression.stochastic_gradient_ascent(feature_matrix, sentiment,
                                                                           initial_coefficients=np.zeros(194),
                                                                           step_size=5e-1, batch_size=1, max_iter=10)

        # Real coefficients that we need to compare with the computed coefficients
        real_coef = [0.26845909, 0.05510662, -0.78232359, 0.24929641, 0.1213813,
                     -0.13194118, -0.42110769, 0.23944013, 0.52334226, 0.30746343,
                     1.46697311, 0.15734639, 0.24112255, -0.22849175, -0.48095714,
                     0., 0.05984944, -0.41942527, -0.48095714, 0.10654088,
                     0., 0.06153186, -0.41942527, 0.43843464,  0.,
                     0.21719583, 0., 0.84326475, 0.28108825, 0.28108825,
                     0., 0., 0.24611428, -0.19986888, 0.15734639,
                     0., 0., -0.48095714, 0.12623269,  0.,
                     0.28108825, 0.07542718, 0., -0.42110769, 0.15734639,
                     -0.48095714, 0.24611428, -0.48095714, 0.,  0.,
                     0.06153186, 0.28108825, 0., 0., 0.,
                     0.05984944, 0.5932902, 0.5621765, -0.48095714, 0.,
                     0.05984944, 0.05984944, 0.31220195, 0.11805882, 0.,
                     0.15085436, 0.24611428, 0., 0., 0.,
                     0.06153186, 0.12623269, 0., 0., 0.,
                     0., 0., 0., -0.35472444, 0.12623269,
                     0.,  0., 0.68023532, 0.28108825, 0.06153186,
                     0.0311137, 0.35651543, 0., 0.28108825, 0.,
                     0.05984944, 0., 0.35651543, 0.28108825, 0.,
                     0., 0., -0.90206483, 0.07542718, -0.48095714,
                     0., 0., -0.48095714, 0., 0.,
                     0., -0.25, 0.0311137, 0., 0.28108825,
                     0., 0., 0., 0., 0.,
                     0., 0.34262011, -0.48095714, 0.28108825, 0.,
                     0., 0., 0., 0., 0.06153186,
                     0.12623269, 0.05984944, 0., 0., 0.,
                     0., 0.12623269, 0., 0., 0.12623269,
                     0.07542718, 0.15085436, 0.07542718, -0.68082602, 0.,
                     0., 0., 0.05984944, 0., 0.,
                     0.28108825, 0., -0.25, 0., 0.,
                     0.07542718, 0., 0., 0.28108825, 0.,
                     0., 0., 0., 0., 0.,
                     0.06153186, 0.0311137, 0., -0.48095714, 0.,
                     0., 0., 0., 0., 0.,
                     0., 0.40732094, 0., 0., 0.05984944,
                     0., 0., 0., 0., 0.,
                     0., 0., 0.06153186, 0., 0.06153186,
                     0., -0.25, 0.05984944, 0., 0.,
                     0., 0., -0.96191427, 0.]

        # Loop through each value, the coefficients must be the same
        for pred_coef, coef in zip(coefficients, real_coef):
            # Assert that both values are the same
            self.assertEqual(round(pred_coef, 5), round(coef, 5))

        # Get the output of the logistic regression with threshold 0
        output = self.predict_output.logistic_regression(feature_matrix, coefficients, 0)

        # Generate a confusion matrix
        confusion_matrix = self.confusion_matrix.confusion_matrix(sentiment, output)

        # Assert the values are to be expected
        self.assertEqual(confusion_matrix, {'false_negatives': 6517, 'true_negatives': 11707,
                                            'true_positives': 17331, 'false_positives': 12225})

        # Assert that the precision is correct
        self.assertEqual(round(self.confusion_matrix.precision(sentiment, output), 5),
                         round(0.72673, 5))

        # Assert that the recall is correct
        self.assertEqual(round(self.confusion_matrix.recall(sentiment, output), 5),
                         round(0.58638, 5))

    def test_02_stochastic_gradient_ascent_high_iteration(self):
        """Test stochastic gradient descent for logistic regression.

        Tests stochastic gradient descent and test it with some known values.

        """
        # We will use important words for the output
        features = self.important_words

        # Output will use sentiment
        output = ['sentiment']

        # Convert our pandas frame to numpy
        feature_matrix, sentiment = self.convert_numpy.convert_to_numpy(self.train_frame, features, output, 1)

        # Compute the coefficients
        coefficients = self.logistic_regression.stochastic_gradient_ascent(feature_matrix, sentiment,
                                                                           initial_coefficients=np.zeros(194),
                                                                           step_size=5e-1, batch_size=1000,
                                                                           max_iter=1000)

        # Real coefficients that we need to compare with the computed coefficients
        real_coef = [-0.06659918,  0.07516305,  0.02337901,  0.91476437,  1.25935729, -0.01093744,
                     -0.29808423,  0.00724611,  1.14319635,  0.58421811, -0.10388794,  0.25341405,
                     0.51935047, -0.16643157,  0.1581433,  -0.01678466,  0.11023426, -0.07801531,
                     -0.11943521, -0.23901842,  0.19961916,  0.26962603,  0.00726172,  1.58116946,
                     -0.04749877, -0.01222728, -0.12452547,  0.2408741,   0.23996495, -0.27318487,
                     0.16391931,  0.46141695, -0.00520781, -0.41720674,  1.3914436,   0.59286041,
                     -0.01877455, -0.1177062,   0.04522629, -0.05050944, -0.1872891,   0.1119123,
                     0.05552736,  0.018883,   -0.28821684,  0.35454167,  0.09146771, -0.15185966,
                     0.45980111,  0.13696004, -0.27719711,  0.37826182,  0.51482099, -0.12707594,
                     -0.08043197,  0.27088589,  0.20836676, -0.22217221,  0.34308818,  0.05011724,
                     0.01336183, -0.00422257,  0.25914879,  0.18971367,  0.11804381,  0.06478439,
                     0.13413068, -0.35940054, -0.04225724, -0.23574987, -0.26178573,  0.37077618,
                     0.266064,    0.0552738,   0.25274691,  0.15248314,  0.9721445,   0.03951392,
                     -0.59577998, -0.09680726, -0.13168621,  0.42806047,  0.03576358,  1.03088019,
                     0.52916025, -0.09516351,  0.23544152,  0.31386904,  0.50647271,  0.25383116,
                     0.1369185,  0.93673001, -0.06280486,  0.1670564,  -0.20573152,  0.2201837,
                     0.12892914, -0.9711816,  -0.24387714, -0.3566874,  -0.65956699, -0.28473646,
                     -0.34083222, -0.44708957, -0.29828401, -0.52797307, -1.92693359, -0.33116364,
                     -0.43025271, -0.21284617, 0.16375567, -0.0299845,  -0.30294927, -1.25019619,
                     -1.55092776, -0.09266983, -0.08014312, -0.07565967, -0.00950432,  0.00327247,
                     0.03190358, -0.04247063, -0.28205865, -0.45678176,  0.06141561, -0.2690871,
                     -0.05979329, -0.0019354,  -0.01279985,  0.05323391, -0.35513613, -0.26639425,
                     -0.41094467, -0.14117863, -0.90001241, -0.33279773,  0.01621988, -0.08709595,
                     -0.10450457, -0.12567406, -0.61727551, -0.18663497,  0.17636203,  0.09316913,
                     -0.06829369,  0.1880183,  -0.5078543,   0.03964466, -0.26089197, -0.07480237,
                     -0.05556211, -0.1450303,  -0.04780934,  0.08911386, -0.15163772,  0.06213261,
                     -0.34512242, -0.33522342,  0.06580618, -0.44499204, -0.68623426, -0.12564489,
                     0.2609755,   0.09998045, -0.25098629, -0.29549973, -0.15944276, -0.47408765,
                     -0.03058168, -1.42253269, -0.49855378,  0.05835175, -1.17789127, -0.08226967,
                     -0.56793665, -0.35814271, -0.98559717, -0.16918106, -0.12477773, -0.23457722,
                     -0.13170106, -0.64351485, -0.01773532, -0.2686544,  0.047442,   -0.34218929,
                     -0.48340895,  0.37866335, -0.25162177,  0.05277577,  0.01545386, -0.26267815,
                     -0.09903819, -0.54500151]

        # Loop through each value, the coefficients must be the same
        for pred_coef, coef in zip(coefficients, real_coef):
            # Assert that both values are the same
            self.assertEqual(round(pred_coef, 5), round(coef, 5))

        # Get the output of the logistic regression with threshold 0
        output = self.predict_output.logistic_regression(feature_matrix, coefficients, 0)

        # Generate a confusion matrix
        confusion_matrix = self.confusion_matrix.confusion_matrix(sentiment, output)

        # Assert the values are to be expected
        self.assertEqual(confusion_matrix, {'false_negatives': 5018, 'true_negatives': 18995,
                                            'true_positives': 18830, 'false_positives': 4937})

        # Assert that the precision is correct
        self.assertEqual(round(self.confusion_matrix.precision(sentiment, output), 5),
                         round(0.78958, 5))

        # Assert that the recall is correct
        self.assertEqual(round(self.confusion_matrix.recall(sentiment, output), 5),
                         round(0.79228, 5))

    def test_04_log_likelihood(self):
        """Test log likelihood.

        Test the log likelihood algorithm, and compare it with some known values.

        """
        # Generate test feature, coefficients, and label
        feature_matrix = np.array([[1., 2., 3.], [1., -1., -1]])
        coefficients = np.array([1., 3., -1.])
        label = np.array([-1, 1])

        # Compute the log likelihood
        lg = self.log_likelhood.log_likelihood(feature_matrix, label, coefficients)

        # Assert the value
        self.assertEqual(round(lg, 5), round(-5.33141161544, 5))

    def test_05_average_log_likelihood(self):
        """Test average log likelihood.

        Test the average log likelihood algorithm, and compare it with some known values.

        """
        # Generate test feature, coefficients, and label
        feature_matrix = np.array([[1., 2., 3.], [1., -1., -1]])
        coefficients = np.array([1., 3., -1.])
        label = np.array([-1, 1])

        # Compute the log likelihood
        lg = self.log_likelhood.average_log_likelihood(feature_matrix, label, coefficients)

        # Assert the value
        self.assertEqual(round(lg, 5), round(-2.6657099999999998, 5))
