"""Implements Node."""


class Node:

    """Base class for other nodes to inherit.

    Attributes:
        inputs (list): A list of nodes that are connected to the current node. For every node in the inputs must be
            inherited with the Node class.
        value (float): Forward propagation's output value. This value will be initially set to None.
        outputs (list): A list of outbound nodes, this will be set by other nodes.
        gradients (dict): A dictionary where the keys are the variables that we will compute the partial respective
            to, where the partial value is the value of the dictionary.

    """

    def __init__(self, inputs):
        """Construct the the Node class. Mainly responsible to set each input's node output to the current node.

        Args:
            inputs (list): A list of nodes that are connected to the current node.

        """
        self.inputs = inputs
        self.value = None
        self.outputs = []
        self.gradients = {}
        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        """Forward propagation function. This function must be used by the node in order to forward propagate."""
        raise NotImplementedError

    def backward(self):
        """Backward propagation function. This function must be used by the node in order to back propagate."""
        raise NotImplementedError
