from unittest import TestLoader, TextTestRunner, TestSuite
from unit_tests.classification.test_logistic_regression import TestLogisticRegression
from unit_tests.classification.test_logistic_regression_l2_norm import TestLogisticRegressionL2Norm
from unit_tests.regression.test_k_nearest_neighbor_regression import TestKNearestNeighborRegression
from unit_tests.regression.test_lasso_regression import TestLassoRegression
from unit_tests.regression.test_linear_regression import TestLinearRegression
from unit_tests.regression.test_ridge_regression import TestRidgeRegression

# Uses a testLoader to run multiple tests from different python unit tests file
if __name__ == "__main__":

    loader = TestLoader()

    suite = TestSuite((
            loader.loadTestsFromTestCase(TestLinearRegression),
            loader.loadTestsFromTestCase(TestRidgeRegression),
            loader.loadTestsFromTestCase(TestLassoRegression),
            loader.loadTestsFromTestCase(TestLogisticRegression),
            loader.loadTestsFromTestCase(TestKNearestNeighborRegression),
            loader.loadTestsFromTestCase(TestLogisticRegressionL2Norm)
        ))

    runner = TextTestRunner()
    runner.run(suite)
