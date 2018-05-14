"""Implements MSE loss function."""

import numpy as np
from .node import Node


class Linear(Node):

    """MSE class for MSE loss function."""

    def __init__(self, y, y_hat):
        """Construct the MSE class. Sets the y (predicted value), y_real (real values) to the input list

        Args:
            y (np.ndarray): Inputs to the node.
            y_hat (np.ndarray): Real outputs.

        """
        Node.__init__(self, inputs=[y, y_hat])

    def forward(self):
        """Perform MSE loss function.

        MSE Loss Function: 1
                           -*∑(y-y_hat)^2
                           M

        """
        # Gather y and y_hat from the input list
        # Reshape so that we can minus two matrices in the same element wise
        y = self.inputs[0].value.reshape(-1, 1)
        y_hat = self.inputs[1].value.reshape(-1, 1)

        # 1
        # -*∑(y-y_hat)^2
        # M
        self.value = np.mean((y - y_hat)**2)

    def backward(self):
        """Compute the gradient through backward propagation.

        Examples used in comments uses the following dimensions:
        x: 2x13
        w: 13x10
        b: 10
        output = 2x10

        Network: Linear => Sigmoid

        """
        # Reshape our inputs so we can minus two matrices element wise
        y = self.inputs[0].value.reshape(-1, 1)
        y_hat = self.inputs[1].value.reshape(-1, 1)

        # inputs[0]=y, partial with respect to the output y
        # Derivative:  dc   d  2
        #              -- = -- -∑(y-y_hat)^2
        #              dy   dy M
        # self.inputs[0] = y = 2x1
        # self.inputs[1] = y_hat = 2x1
        # self.inputs[0] - self.inputs[1] = 2x1
        # Cost with respective to X for the network:
        # dc   dc  ds  dnet
        # -- = -- ---- ----, where s is the sigmoid function
        # dw   ds dnet  dw
        #                  s_1               net_1
        #            |-------------|        |-----|
        #      2           1            e^(w_1*x_1)
        #    = -(y - --------------)(-------------------)(  x_1  )
        #      M     1+e^-(w_1*x_1)  (1 + e^(w_1*x_1))^2
        #       |-------------------||-------------------||-------|
        #                 dc                  ds_1         dnet_1
        #                ----                ------        ------
        #                ds_1                dnet_1         dx_1
        # 2x1 (gradients for y) = 2/M*(y(2x1)-y_hat(2x1)) These are the gradients passed down to y
        self.gradients[self.inputs[0]] = (2 / self.inputs[0].value.shape[0]) * (y-y_hat)

        # inputs[0]=y, partial with respect to the output y
        # Derivative:  dc   d   2
        #              -- = -- --∑(y-y_hat)^2
        #              dy   dy  M
        # self.inputs[0] = y = 2x1
        # self.inputs[1] = y_hat = 2x1
        # self.inputs[0] - self.inputs[1] = 2x1
        # Cost with respective to X for the network:
        # dc   dc  ds  dnet
        # -- = -- ---- ----, where s is the sigmoid function
        # dw   ds dnet  dw
        #                    s_1               net_1
        #              |-------------|        |-----|
        #        2           1            e^(w_1*x_1)
        #    =  --(y - --------------)(-------------------)(  x_1  )
        #        M     1+e^-(w_1*x_1)  (1 + e^(w_1*x_1))^2
        #         |-------------------||-------------------||-------|
        #                   dc                  ds_1         dnet_1
        #                  ----                ------        ------
        #                  ds_1                dnet_1         dx_1
        # 2x1 (gradients for y) = -2/M*(y(2x1)-y_hat(2x1)) These are the gradients passed down to y hat
        self.gradients[self.inputs[1]] = (-2 / self.inputs[0].value.shape[0]) * (y-y_hat)
