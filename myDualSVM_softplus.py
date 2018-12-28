import sys

import numpy as np
from cvxopt import matrix, solvers

from util import load_data


class DualSVM(object):
    """Binary SVM classifier using the Lagrangian primal objective and dual.

    Attributes
    ----------
    C: double
        The regularization parameter.
    w: numpy.ndarray
        The trained weights vector.
    b: float
        The intercept.
    n_support_vectors: int
        The number of support vectors.
    margin: double
        The size of the margin.

    """

    def __init__(self, C):
        self.C = C
        self.w = None
        self.b = None
        self.n_support_vectors = None
        self.margin = None

    def fit(self, X, y):
        """Trains a SVM using optimization of the Lagrangian dual.

        Parameters
        ----------
        X: numpy.ndarray
            Data points.
        y: numpy.ndarray

        Returns
        -------
        float
            Training error rate.

        """
        n = X.shape[0]
        yx = y[:, np.newaxis] * X

        # the lagrangian dual
        P = yx @ yx.T
        q = -np.ones((n, 1))

        # inequality constraints x <= C (top half) and -x <= 0 (then x >= 0) (bottom half)
        G = np.vstack((np.identity(n), -np.identity(n)))  # left side multipliers on x
        h = np.vstack((self.C * np.ones((n, 1)), np.zeros((n, 1))))  # right side

        # equality constraints, inner product of target and x is 0, solutions of weights and intercepts
        A = y[np.newaxis, :]  # 1 x N
        b = np.zeros((1, 1))

        solution = solvers.qp(*(matrix(arg) for arg in (P, q, G, h, A, b)))

        alphas = np.array(solution['x'])

        max_support_vector = alphas.max()
        tolerance = max_support_vector * 1e-3  # All alphas that are withing 3 orders of magnitude of the the maximum
        support_vectors = ~np.isclose(alphas, 0, atol=tolerance).ravel()
        self.n_support_vectors = np.count_nonzero(support_vectors)
        self.w = (alphas[support_vectors] * yx[support_vectors]).sum(axis=0)

        # Use the closest support vectors on each side of the hyperplane to compute b, taking the average
        self.b = - 0.5 * (np.max(X[y == -1] @ self.w) + np.min(X[y == 1] @ self.w))

        self.margin = 1 / np.linalg.norm(self.w)
        return np.count_nonzero(self.predict(X) != y) / n

    def predict(self, X):
        """Predicts a set of data points using previously trained weights and intercept.

        Parameters
        ----------
        X: numpy.ndarray
            The data points to predict

        Returns
        -------
        float
            -1 predicting the -1 class, 1 predicting the 1 class
        """
        return np.sign(X @ self.w + self.b)


def cross_validation(model, X, y, k=10):
    """Performs cross validation using k splits and all combinations of two of the splits as test data.

    Parameters
    ----------
    model: DualSVM
        The SVM model.
    X: numpy.ndarray
        The data points.
    y: numpy.ndarray
        The targets, -1 or 1
    k: int
        The number of splits.

    Returns
    -------
    dict
        A dictionary with the following keys:
            n_support_vectors - the number of support vectors for each run.
            margins - The size of the margin for each run
            train_error_rates - The train error rate for each run
            test_error_rates - The test error rate for each run

    """
    data = np.hstack((X, y.reshape(-1, 1)))
    data_splits = np.array_split(data, 10, axis=0)

    results = {
        'n_support_vectors': [],
        'margins': [],
        'train_error_rates': [],
        'test_error_rates': [],
    }
    it = 0
    for i in range(k):
        for j in range(i + 1, k):
            solvers.options['show_progress'] = False

            train = np.concatenate([data_split for x, data_split in enumerate(data_splits) if x != i and x != j])
            train_error_rate = model.fit(train[:, :-1], train[:, -1])

            results['n_support_vectors'].append(model.n_support_vectors)
            results['margins'].append(model.margin)
            results['train_error_rates'].append(train_error_rate)

            test = np.vstack((data_splits[i], data_splits[j]))
            prediction = model.predict(test[:, :-1])
            errors = np.count_nonzero(prediction != test[:, -1])
            error_rate = errors / test.shape[0]
            results['test_error_rates'].append(error_rate)
            print(f"Iteration: {it} support_vectors: {model.n_support_vectors} margin: {model.margin:.3f} "
                  f"train: {train_error_rate:.3f} test: {error_rate:.3f}", end='\r')
            it += 1

    print('')
    return results


def main(file, C):
    """Main method for compatibility with instructions, I used a jupyter notebook, not this.
    Performs SVM training and testing for a value of C.

    Parameters
    ----------
    file: str
        The path to the file to load.
    C: double
        The regularization parameter C for the SVM.

    """
    X, y = load_data(file)
    svm = DualSVM(C)
    results = cross_validation(svm, X, y)
    print("C:", C)
    print("n_support_vectors mean:", np.mean(results['n_support_vectors']))
    print("n_support_vectors std:", np.std(results['n_support_vectors']))
    print("margins mean:", np.mean(results['margins']))
    print("margins std:", np.std(results['margins']))
    print("training error mean:", np.mean(results['train_error_rates']))
    print("training error std:", np.std(results['train_error_rates']))
    print("test error mean:", np.mean(results['test_error_rates']))
    print("test error std:", np.std(results['test_error_rates']))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))
