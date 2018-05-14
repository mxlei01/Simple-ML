"""Implements NearestNeighbor Unittest."""

import unittest
import numpy as np
from sklearn.datasets import load_boston
from machine_learning.neural_networks.input import Input
from machine_learning.neural_networks.linear import Linear
from machine_learning.neural_networks.mse import Mse
from machine_learning.neural_networks.sigmoid import Sigmoid
from machine_learning.neural_networks.compute import topological_sort, update_network


class TestNeuralNetwork(unittest.TestCase):

    """Tests for TestNeuralNetwork.

    Statics:
        _multiprocess_can_split_ (bool): Flag for nose tests to run tests in parallel.

    """

    _multiprocess_can_split_ = True

    def setUp(self):
        """Set up for TestNeuralNetwork.

        Sets up a basic neural network for testing.

        """
        # Load data
        data = load_boston()
        self.x_ = data['data']
        self.y_ = data['target']

        # Normalize data
        self.x_ = (self.x_ - np.mean(self.x_, axis=0)) / np.std(self.x_, axis=0)

        np.random.seed(0)
        self.n_features = self.x_.shape[1]
        self.n_hidden = 10
        self.w1_ = np.random.randn(self.n_features, self.n_hidden)
        self.b1_ = np.zeros(self.n_hidden)
        self.w2_ = np.random.randn(self.n_hidden, 1)
        self.b2_ = np.zeros(1)

        # Neural network
        self.x, self.y = Input(), Input()
        self.w1, self.b1 = Input(), Input()
        self.w2, self.b2 = Input(), Input()

        self.l1 = Linear(self.x, self.w1, self.b1)
        self.s1 = Sigmoid(self.l1)
        self.l2 = Linear(self.s1, self.w2, self.b2)
        self.cost = Mse(self.y, self.l2)

        self.feed_dict = {
            self.x: self.x_,
            self.y: self.y_,
            self.w1: self.w1_,
            self.b1: self.b1_,
            self.w2: self.w2_,
            self.b2: self.b2_
        }

    def test_01_training(self):
        """Test neural network training."""
        # Setup our parameters
        epochs = 10
        m = self.x_.shape[0]
        batch_size = 2
        steps_per_epoch = m // batch_size

        # Create our sorted graph
        graph = topological_sort(self.feed_dict)

        # Declare our trainables, so that we know which values can be updated
        trainables = [self.w1, self.b1, self.w2, self.b2]

        # Train our neural network
        update_network(graph, self.x, self.y, self.x_, self.y_, trainables, epochs, steps_per_epoch,
                       batch_size, verbose=False)

        # Make sure the bias values are the same, not required to test other weight
        self.assertEqual(np.round(self.b2.value[0], 5), np.round(5.68319026, 5))
