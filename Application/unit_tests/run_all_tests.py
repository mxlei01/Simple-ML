from unit_tests.test_linear_regression import TestLinearRegression
from unittest import TestLoader, TextTestRunner, TestSuite

# Uses a testLoader to run multiple tests from different python unit tests file
if __name__ == "__main__":

    loader = TestLoader()

    suite = TestSuite((
            loader.loadTestsFromTestCase(TestLinearRegression)
        ))

    runner = TextTestRunner()
    runner.run(suite)
