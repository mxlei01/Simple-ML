import numpy as np

class ResidualSumSquares():
    # Usage:
    #   Computes residual sum of squares

    def residual_sum_squares_linear_regression(self, output, predicted_output):
        # Usage:
        #   Computes for residual sum of squares for linear regression, which computes
        #   RSS = (y-Hw)^t * (y-Hw)
        # Arguments
        #   output           (numpy array) : real output
        #   predicted_output (numpy array) : predicted output
        # Return:
        #   (y-Hw)^t * (y-Hw)

        return np.dot(output-predicted_output, output-predicted_output)
