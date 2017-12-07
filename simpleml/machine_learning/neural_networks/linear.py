"""Implements linear transform."""
from .node import Node
import numpy as np


class Linear(Node):

    """Linear class for linear transformation."""

    def __init__(self, x, w, b):
        """Construct the Linear class. Sets the x (input), w (weight), and b (bias), value to the inputs list.

        Args:
            x (np.ndarray): Inputs to the node.
            w (np.ndarray): Weights to the node.
            b (np.array): Biases to the Node.

        """
        Node.__init__(self, inputs=[x, w, b])

    def forward(self):
        """Performs linear transform.

        Linear Transform: ∑(x_i*w_i)+b, where i is the number of x (inputs), and w (weights).


        """
        # Gather x, w, and b from the input list
        x = self.inputs[0].value
        w = self.inputs[1].value
        b = self.inputs[2].value

        # ∑(x_i*w_i)+b
        self.value = np.dot(x, w) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)