from unit_tests.test_linear_regression import TestLinearRegression
from unit_tests.test_ridge_regression import TestRidgeRegression
from unit_tests.test_lasso_regression import TestLassoRegression
from unit_tests.test_logistic_regression import TestLogisticRegression
from unit_tests.test_k_nearest_neighbor_regression import TestKNearestNeighborRegression
from unittest import TestLoader, TextTestRunner, TestSuite

# Uses a testLoader to run multiple tests from different python unit tests file
if __name__ == "__main__":

    loader = TestLoader()

    suite = TestSuite((
            loader.loadTestsFromTestCase(TestLinearRegression),
            loader.loadTestsFromTestCase(TestRidgeRegression),
            loader.loadTestsFromTestCase(TestLassoRegression),
            loader.loadTestsFromTestCase(TestLogisticRegression),
            loader.loadTestsFromTestCase(TestKNearestNeighborRegression)
        ))

    runner = TextTestRunner()
    runner.run(suite)
