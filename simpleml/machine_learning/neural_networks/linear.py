"""Implements linear transform."""

import numpy as np
from .node import Node


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
        """Perform linear transform.

        Linear Transform: ∑(x_i*w_i)+b, where i is the number of x (inputs), and w (weights).

        """
        # Gather x, w, and b from the input list
        x = self.inputs[0].value
        w = self.inputs[1].value
        b = self.inputs[2].value

        # ∑(x_i*w_i)+b
        self.value = np.dot(x, w) + b

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
        # For each inputs: x, w, b, we need to find their respective gradient values.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        # The gradients are summed over all the outputs. If a linear node connects to multiple activation functions,
        # then each of the activation function would have it's loss respective to the this node. We can get each
        # loss, and compute the loss associated with each input function.
        for n in self.outputs:
            # Partial of the cost with respective to this node from the output node. This is the cumulative derivative
            # until (not including this node)'s cost. For example, according to our network, this is:
            # dc  ds
            # -- ---- = 2x10 matrix
            # ds dnet
            # Our next task is to multiply this matrix with either w or x to get the cost with respect to x or w.
            grad_cost = n.gradients[self]

            # inputs[0]=x, partial of the loss respective to this node's input.
            # Derivative:  dl   d
            #              -- = --(∑w_i*x_i+b)=w_i
            #              dx   dx
            # self.outputs[1].value.T = w.T = 10x13
            #        dc  ds
            # np.dot(-- ----(grad_cost), w.T) = 2x13, this 2x13 produces the cost for the inputs of 2x13
            #        ds dnet
            # Cost with respective to X for the network:
            # dc   dc  ds  dnet
            # -- = -- ---- ----, where s is the sigmoid function
            # dx   ds dnet  dx
            #                  s_1               net_1
            #            |-------------|        |-----|
            #                   1            e^(w_1*x_1)
            #    = -(y - --------------)(-------------------)(  w_1  )
            #            1+e^-(w_1*x_1)  (1 + e^(w_1*x_1))^2  
            #      |-------------------||-------------------||-------|    
            #                dc                  ds_1         dnet_1
            #               ----                ------        ------
            #               ds_1                dnet_1         dx_1
            # 2x13 (gradients for x) = 2x10 (grad_cost) * 10x13 (w.T), if the inputs are from training data, then
            # we will do nothing with the gradients, however, if the gradients are another layer of activation with
            # weights, then we can pass the cost downwards.
            self.gradients[self.inputs[0]] += np.dot(grad_cost, self.outputs[1].value.T)

            # inputs[1]=w, partial of the loss respective to this node's output
            # Derivative: dl   d
            #             -- = --(∑w_i*x_i+b)=x_i
            #             dw   dw
            # self.outputs[0].value.T = x.T = 13x2
            #             dc  ds
            # np.dot(x.T, -- ----(grad_cost)) = 13x10, this 13x10 produces the cost for the weights of 13x10
            #             ds dnet
            # Cost with respective to W for the network:
            # dc   dc  ds  dnet
            # -- = -- ---- ----, where s is the sigmoid function
            # dw   ds dnet  dw
            #                  s_1               net_1
            #            |-------------|        |-----|
            #                   1            e^(w_1*x_1)
            #    = -(y - --------------)(-------------------)(  w_1  )
            #            1+e^-(w_1*x_1)  (1 + e^(w_1*x_1))^2  
            #      |-------------------||-------------------||-------|    
            #                dc                  ds_1         dnet_1
            #               ----                ------        ------
            #               ds_1                dnet_1         dx_1
            # 13x10 (gradients for w) = 13x2 (x.T) * 2*10 (grad_cost), this is the gradient values for w, and we can
            # use the gradient values to update w.
            self.gradients[self.inputs[1]] += np.dot(self.outputs[0].value.T, grad_cost)

            # inputs[2]=b, partial of the loss respective to this node's bias
            # Derivative: dl   d
            #             -- = --(∑w_i*x_i+b)=1
            #             db   dw
            # np.sum(grad_cost) = 1x10 matrix
            # Cost with respective to b for the network:
            # dc   dc  ds  dnet
            # -- = -- ---- ----, where s is the sigmoid function
            # db   ds dnet  db
            #                  s_1               net_1
            #            |-------------|        |-----|
            #                   1            e^(w_1*x_1)
            #    = -(y - --------------)(-------------------)(1)
            #            1+e^-(w_1*x_1)  (1 + e^(w_1*x_1))^2
            #      |-------------------||-------------------||-------|
            #                dc                  ds_1         dnet_1
            #               ----                ------        ------
            #               ds_1                dnet_1          db
            # 1x10 (gradients for b) = 2x10 with columns summed together, this is the gradient values for b, and we can
            # use the gradient values to update b.
            self.gradients[self.inputs[2]] += np.sum(grad_cost, axis=0, keepdims=False)
