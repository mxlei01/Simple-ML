from unit_tests.test_linear_regression import TestLinearRegression
from unit_tests.test_ridge_regression import TestRidgeRegression
from unittest import TestLoader, TextTestRunner, TestSuite

# Uses a testLoader to run multiple tests from different python unit tests file
if __name__ == "__main__":

    loader = TestLoader()

    suite = TestSuite((
            loader.loadTestsFromTestCase(TestLinearRegression),
            loader.loadTestsFromTestCase(TestRidgeRegression)
        ))

    runner = TextTestRunner()
    runner.run(suite)
