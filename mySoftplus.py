import json
import sys

import numpy as np

from util import Model, load_data


class SoftplusSGD(Model):
    """Trains a softplus_a classifier using SGD/GD

    Parameters
    ----------
    initial_w: numpy.ndarray
        An initial weights vector.
    lr: double
        The learning rate.
    l: double
        The regularization parameter.
    a: double
        The softplus_a parameter, smaller values mean closer to real hinge loss.

    Attributes
    ----------
    w: numpy.ndarray
        The current weights.
    w_avg: numpy.ndarray
        A moving average of all computed weights.
    comps: int
        Number of computations.
    """

    def __init__(self, initial_w, lr=0.05, l=0.0001, a=0.05):
        self.w = initial_w
        self.lr = lr
        self.l = l
        self.a = a
        self.comps = 1
        self.w_avg = initial_w

    def _loss(self, X, y, w):
        """Softplus objective loss.

        Parameters
        ----------
        X: numpy.ndarray
            The data points.
        y: numpy.ndarray
            The targets, -1 or 1.
        w: numpy.ndarray
            The weights to use to compute loss.

        Returns
        -------
        double
            The loss value.

        """
        L = self.a * np.log(1 + np.exp((1 - y[:, np.newaxis] * X @ w) / self.a))
        if len(L.shape) > 0:
            return L.mean(axis=0) + self.l * np.linalg.norm(w) ** 2
        else:  # handle k = 1, where we don't compute a mean
            return L + self.l * np.linalg.norm(w) ** 2

    def loss(self, X, y):
        """Computes the softplus_a loss on a set of data points and their targets using the cumulative moving average of
        the weight history.

        Parameters
        ----------
        X: numpy.ndarray
            The data points.
        y: numpy.ndarray
            The targets, -1 or 1.

        Returns
        -------
        float
            The loss value.


        """
        w = self.w_avg
        return self._loss(X, y, w)

    def _batch_loss(self, X, y):
        """A partial loss computed using the current weight value.

        Parameters
        ----------
        X: numpy.ndarray
            The data points.
        y: numpy.ndarray
            The targets, -1 or 1.

        Returns
        -------
        float
            The loss value.

        """
        w = self.w
        return self._loss(X, y, w)

    def gradient(self, X, y):
        """The gradient computed using the current weight value.

        Parameters
        ----------
        X: numpy.ndarray
            The data points.
        y: numpy.ndarray
            The targets, -1 or 1.

        Returns
        -------
        numpy.ndarray
            The gradient of the weights.

        """
        y = y[:, np.newaxis]
        e = np.exp((1 - y * X @ self.w) / self.a)[:, np.newaxis]
        s = e * - y * X / (1 + e)
        if len(s.shape) > 1:
            return s.mean(axis=0) + 2 * self.l * self.w
        else:  # handle k = 1, where a mean is not needed and will fail.
            return s + 2 * self.l * self.w

    def fit_batch(self, X, y):
        """Fits a batch, computing the gradient and adjusting weights based on that gradient.

        Parameters
        ----------
        X: numpy.ndarray
            The data points.
        y: numpy.ndarray
            The targets, -1 or 1.

        Returns
        -------
        float
            A partial loss computed on only the batch, with

        """
        g = self.gradient(X, y)
        self.w = self.w - self.lr * g
        self.w_avg = self.w_avg + (self.w - self.w_avg) / (self.comps + 1)  # cumulative moving average of weights
        self.comps += 1
        return self._batch_loss(X, y)

    def predict(self, X):
        """Predicts the classes of data points using the average weights.

        Parameters
        ----------
        X: numpy.ndarray
            N x M data points matrix.

        Returns
        -------
        numpy.ndarray
            N x 1 predictions matrix.

        """
        return X @ self.w_avg


def main(filename, k, numruns):
    """Main method for compatibility with instructions, I used a jupyter notebook, not this.

    Parameters
    ----------
    filename
    k
    numruns

    """
    X, y = load_data(filename)
    sp = SoftplusSGD(initial_w=np.random.random(X.shape[1]) * 1e-5)
    seconds = []
    objective_values = []
    for i in range(numruns):
        print("Run ", i)
        tt, _, os = sp.train(X, y, k=k)
        seconds.append(tt)
        objective_values.append(os)

    with open("softplus_out.json", "w") as f:
        json.dump({'seconds': seconds, 'objective_values': objective_values}, f)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
