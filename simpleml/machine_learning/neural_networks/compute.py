"""Functions to run a neural network."""

from sklearn.utils import resample
from .input import Input


def topological_sort(feed_dict):
    """Topological sort in order to know which nodes to run.

    Args:
        feed_dict (dict): A dictionary of node input values.

    Returns:
        sorted_nodes (list): A list of sorted nodes.

    """
    input_nodes = [n for n in feed_dict.keys()]

    graph = {}
    nodes = [n for n in input_nodes]
    while nodes:
        n = nodes.pop(0)
        if n not in graph:
            graph[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in graph:
                graph[m] = {'in': set(), 'out': set()}
            graph[n]['out'].add(m)
            graph[m]['in'].add(n)
            nodes.append(m)

    sorted_nodes = []
    set_nodes = set(input_nodes)
    while set_nodes:
        n = set_nodes.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        sorted_nodes.append(n)
        for m in n.outputs:
            graph[n]['out'].remove(m)
            graph[m]['in'].remove(n)
            if not graph[m]['in']:
                set_nodes.add(m)
    return sorted_nodes


def forward_and_backward(graph):
    """Forward pass and a backward pass.

    Args:
        graph (list): Topological sorted graph in a list.

    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """Stochastic gradient descent.

    Args:
        trainables (list of Input): A list of trainable nodes (that has a weight).

    """
    for t in trainables:
        t.value = t.value - learning_rate * t.gradients[t]


def update_network(graph, x, y, x_, y_, trainables, epochs, steps_per_epoch, batch_size, verbose=True):
    """Update a neural network.

    Args:
        graph (list): Sorted neural network (graph) as a list.
        x (Input): A node for the input.
        y (Input): A node for the output.
        x (numpy.ndarray): A matrix of inputs.
        y (numpy.ndarray): A list of real outputs.
        trainables (list of Input): A list of trainable nodes (that has a weight).
        epochs (int): Number of epochs to train the neural network.
        steps_per_epoch (int): Sample steps per epoch.
        batch_size (int): The number of batch size to train.
        verbose (bool): Displays messages during training.

    """
    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # Sample a batch of data
            x_batch, y_batch = resample(x_, y_, n_samples=batch_size, random_state=j)

            # Reset the node input
            x.value = x_batch
            y.value = y_batch

            # Forward and backward propagate
            forward_and_backward(graph)

            # Use SGD to update our weights
            sgd_update(trainables)

            # Increment our loss
            loss += graph[-1].value

        if verbose:
            print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
