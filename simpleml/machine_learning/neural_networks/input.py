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

        # If self is a weight or bias node, then we need to add up all the gradients that belongs to "self" from the 
        # the list of self's output.
        # For example: w ==> l_1 => s_1
        #              | ==> l_2 => s_2
        # w (weight) node connects to l_1 (linear) and l_2 (linear) nodes, and l_1 connects to s_1 (sigmoid), and l_2
        # connects to s_2 (sigmoid).
        # Both l_1 and l_2 and will contain gradients for w node.
        # Then, dc
        #       -- = dl_1 + dl_2
        #       dw
        # If we only consider l_1 at the moment, then we can expand l_1:
        # dc   dc  ds  dnet  where dnet is the linear combination that occurs in the sigmoid function (input*weights).
        # -- = -- ---- ---- 
        # dw   ds dnet  dw 
        #                                  |-net-|
        #                   1           e^(w_1*x_1)
        #    = -(y - --------------)(-----------------)(x_1)
        #            1+e^-(w_1*x_1)  (1+e^(w_1*x_1))^2 
        #            |------s-----|                    
        #               dc                 ds          dnet
        #               --                ----         ----
        #               ds                dnet          dw
        # Hence when the derivate of n.gradients[self] would be the value of dc
        #                                                                    --
        #                                                                    dw
        # If we have more than one linear connection, then we would need to sum them up, such as dl_1 + dl_2
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]