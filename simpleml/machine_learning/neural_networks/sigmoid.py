"""Implements sigmoid function."""

import numpy as np
from .node import Node


class Sigmoid(Node):

    """Sigmoid class for sigmoid function."""

    def __init__(self, x):
        """Construct the sigmoid class. Sets the x (input) value to the inputs list.

        Args:
            x (np.ndarray): Inputs to the node.

        """
        Node.__init__(self, inputs=[x])

    def forward(self):
        """Perform sigmoid function.

        Sigmoid Function:   1
                         --------
                         1 + e^-x

        """
        # Gather x, w, and b from the input list
        x = self.inputs[0].value

        #    1
        # --------
        # 1 + e^-x
        self.value = 1. / (1. + np.exp(-x))

    def backward(self):
        """Compute the gradient through backward propagation.

        Examples used in comments uses the following dimensions:
        x: 2x13
        w: 13x10
        b: 10
        output = 2x10

        Network: Linear => Sigmoid

        """
        # Initialize a partial for each of the inbound_nodes.
        # For x we need to find their respective gradient values.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        # The gradients are summed over all the outputs. If a sigmoid node connects to multiple activation functions,
        # then each of the activation function would have it's loss respective to the this node. We can get each
        # loss, and compute the loss associated with each input function.
        for n in self.outputs:
            # Partial of the cost with respective to this node from the output node. This is the cumulative derivative
            # until (not including this node)'s cost. For example, according to our network, this is:
            # dc
            # -- = 2x10 matrix
            # ds
            grad_cost = n.gradients[self]

            # Assign a different name to self.value
            sigmoid = self.value

            # inputs[0]=x, partial of the loss respective to this node's output
            # Derivative: dl
            #             -- = sigmoid * (1 - sigmoid)
            #             dsd
            # Cost with respective to W for the network:
            # dc   dc  ds  dnet
            # -- = -- ---- ----, where s is the sigmoid function
            # dw   ds dnet  dw
            #                  s_1               net_1
            #            |-------------|        |-----|
            #                   1            e^(w_1*x_1)
            #    = -(y - --------------)(-------------------)(  x_1  )
            #            1+e^-(w_1*x_1)  (1 + e^(w_1*x_1))^2
            #      |-------------------||-------------------||-------|
            #                dc                  ds_1         dnet_1
            #               ----                ------        ------
            #               ds_1                dnet_1         dw_1
            # Note: x_1 will be done in the linear node
            # 2x10 (gradients for w) = 2x10 (x.T) .(dot product) 2x10 (grad_cost), each value in self.inputs[0], loss,
            # are multiplied with the grad_cost element wise.
            self.gradients[self.inputs[0]] += sigmoid * (1 - sigmoid) * grad_cost
