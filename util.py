import abc
from time import perf_counter

import numpy as np


def load_data(file, class_values=None):
    """Loads the MNIST13 data set and normalizes it.

    Parameters
    ----------
    file
    class_values

    Returns
    -------

    """
    if class_values is None:
        class_values = [-1., 1.]
    data = np.genfromtxt(file, delimiter=',')
    X = data[:, 1:]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = data[:, 0]
    classes = np.unique(y)
    if classes.shape[0] != 2:
        raise ValueError("Needs to have exactly two classes.")
    y = np.select([y == classes[0], y == classes[1]], class_values)
    return X, y


def _batches(data, batch_size, shuffle):
    """A generator function for batching.

    Parameters
    ----------
    data : numpy.ndarray
        A N x M data array with samples on axis 0 and features and targets on axis 1. Columns 0 to M-1 exclusive are the
        features, and column M-1 is the targets.
    batch_size : int
        The size of
    shuffle

    Returns
    -------

    """
    samples = data.shape[0]

    if batch_size == samples:
        # yield the entire set of data
        # shuffling is pointless here since gradients are going to be computed against all the data.
        while True:
            yield data[:, :-1], data[:, -1]
    elif batch_size == 1:
        # yield one data point at a time
        while True:
            if shuffle:
                np.random.shuffle(data)

            for i in range(data.shape[0]):
                yield data[i, :-1], data[i, -1, np.newaxis]
    else:
        classes = np.unique(data[:, -1])
        classes_data = [data[data[:, -1] == cls] for cls in classes]
        n_splits = samples // batch_size

        while True:
            if shuffle:
                for class_data in classes_data:
                    np.random.shuffle(class_data)

            # we flip odds here because the classes individually may not be divisible by the number of splits, and
            # array_split stacks the arrays with more towards one side.
            classes_splits = [np.array_split(class_data, n_splits) if i % 2 == 0
                              else np.flip(np.array_split(class_data, n_splits), axis=0)
                              for i, class_data in enumerate(classes_data)]
            for i in range(n_splits):
                all_class_splits = [class_split[i] for class_split in classes_splits]  # i-th split from all classes
                concatenated = np.concatenate(all_class_splits, axis=0)  # concatenate the i-th split of all classes
                yield concatenated[:, :-1], concatenated[:, -1]  # split into X and y


class Model(abc.ABC):
    """An abstract base class model for doing the question 2 batch training process.
    """
    def fit_batch(self, X, y):
        """Fits a batch of data, adjusting weights based on the gradient computed from that batch.

        Parameters
        ----------
        X : numpy.ndarray
            Data points.
        y : numpy.ndarray
            Targets.

        Returns
        -------
        double
            The loss counterpart to the gradient that was computed.

        """
        raise NotImplementedError

    def loss(self, X, y):
        """Computes the objective loss on a set of data with weights as though the training process would end right now.

        Parameters
        ----------
        X : numpy.ndarray
            Data points.
        y : numpy.ndarray
            Targets.

        Returns
        -------
        double
            The loss computed the data points and targets.

        """
        raise NotImplementedError

    def train(self, X, y, k=1, ktot=None, shuffle=True):
        """

        Parameters
        ----------
        X : numpy.ndarray
            input data, data points are rows, features are columns
        y : numpy.ndarray
            input targets, 1 target class per data point
        k : int
            Number of samples per batch
        ktot : int
            Total number of gradient computations. Default will use 100 times the number of samples
        shuffle : bool
            Whether to shuffle the data in-between epochs.

        Returns
        -------
        tuple
            Tuple of time taken in seconds as a double, steps for which we have the objective

        """
        if ktot is None:
            ktot = 100 * X.shape[0]
        data = np.hstack((X, y[:, np.newaxis]))
        batches = _batches(data, k, shuffle)
        old_seconds = 0.
        seconds = 0.
        steps = []
        objective_values = []
        i = 0
        while i < ktot:
            X_, y_ = next(batches)
            if k > 1 and np.unique(y).shape[0] < 2:
                print(y)
            start = perf_counter()
            partial_loss = self.fit_batch(X_, y_)
            stop = perf_counter()
            seconds += stop - start
            if seconds - old_seconds > .1:  # Only compute the loss on the total data set every .1 seconds at most
                objective_value = self.loss(X, y)
                steps.append(i)
                objective_values.append(objective_value)
                print(f"Iter [{i}] batch loss: {partial_loss:.3f} total loss: {objective_value:.3f}", end='\r')
                old_seconds = seconds
            i += 1

        objective_value = self.loss(X, y)
        steps.append(i)
        objective_values.append(objective_value)
        print(f"Iter [{i}] total loss: {objective_value:.3f}                                       ")

        return seconds, steps, objective_values
