"""Implements Input."""
from .node import Node


class Input(Node):

    """Input class for inputs to use."""

    def __init__(self):
        """Construct the Input node class. Does not need to do anything special.

        The value variable will be set by topological sort later.

        """
        Node.__init__(self, inputs=[])

    def forward(self):
        """Forward propagation function. Does not require to forward propagate anything."""
        pass

    def backward(self):
        """Backward propagation function.

        Inputs, which are not weights or biases does not have gradients, so it is set to 0.

        The weight and bias requires the gradients to be summed up, since the weight and bias node would utilize the
        input node.

        """
        self.gradients = {self: 0}

        # Requires us to add the
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]
