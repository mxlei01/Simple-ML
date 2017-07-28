"""Implements NormalizeFeatures."""

import numpy as np


class NormalizeFeatures:

    """For normalizing a numpy matrices.

    The NormalizeFeatures class contains useful functions to normalize features, for example, applying
    L2 Norm to a matrix.

    """

    @staticmethod
    def l2_norm(features_matrix):
        """Normalize a numpy matrix with l2 norm.

        Normalize each column of a numpy matrix with l2 norm: http://mathworld.wolfram.com/L2-Norm.html
        also called Euclidean Norm: sqrt(x^2+y^2+z^2).

        Args:
            features_matrix(numpy.ndarray): Numpy matrix to normalize.

        Returns:
            A tuple that contains a L2 normalized numpy matrix, and the norm:
                (
                    features_numpy (numpy.matrix): A l2 normalized numpy matrix.
                    norm (list of float): L2 Norms used to normalize each numpy column.
                )

        """
        # Compute the 2-norm of each column
        # [norm(X[:,0]), norm(X[:,1]), norm(X[:,2])]
        # Note that for norm, axis=0->column
        #                     axis=1->row
        norms = np.linalg.norm(features_matrix, axis=0)

        # Compute the norm of each column by column/2-norm of column
        # [X[:,0]/norm(X[:,0]), X[:,1]/norm(X[:,1]), X[:,2]/norm(X[:,2])]
        return features_matrix / norms, norms
