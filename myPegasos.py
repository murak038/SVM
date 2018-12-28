import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt



def separateClasses(classes, values):
    # this function classifies data corresponding to 1 as -1.0
    # and target values corresponding to 3 as 1.0

    target = np.zeros([values.shape[0], 1])
    for i in range(0, target.shape[0]):
        if values[i] == classes[0]:
            target[i] = -1.0
        else:
            target[i] = 1.0
    return target


class Pegasos(object):
    def __init__(self, data, target, l, k, T):
        self.l = l
        self.k = k
        self.T = T
        self.classes = np.unique(target)

    def train(self, data, target):
        # this function is used to train the Pegasos SVM

        w = np.zeros([1, data.shape[1]])  # initialize weight vector as 0s
        k_tot = 0  # initialize total number of runs
        loss = []  # initalize loss vector
        t = 0  # initialize run variable
        while (t < self.T) and (k_tot <= 100 * data.shape[0]):
            A_data, A_target = self.sample(data, target)  # select a subset of the data of k points

            # find values of A that correspond to Ay*dot(Ax,weight) < 1
            index = np.arange(0, A_data.shape[0])
            inde = index[((A_target * np.dot(A_data, w.T)) < 1).reshape(index.shape)]
            A_plus_data = A_data[inde, :]
            A_plus_target = A_target[inde, :]
            # finds the learning rate at the step
            if t == 0:
                lr = 1.0 / (self.l)
            else:
                lr = 1.0 / (self.l * t)
            # stochastic subgradient descent step
            w_half = (1. - lr * self.l) * w + (lr / self.k) * np.sum(np.multiply(A_plus_target, A_plus_data), axis=0)
            # projection step
            w = np.minimum(np.float64(1.0), 1.0 / (np.sqrt(self.l) * np.linalg.norm(w_half))) * w_half
            # add the loss to the vector
            loss.append(self.cost_function(data, target, w))
            # stop the training if the loss of the function decreases
            # below the threshold (1e-5)
            if loss[t] < (1e-3):
                break
            k_tot += 1
            t += 1
        return loss

    def sample(self, data, target):
        # this function samples k random points from both classes
        index = np.arange(0, data.shape[0]).reshape([data.shape[0], 1])
        index_0 = index[target == self.classes[0]]  # index of points of first class
        index_1 = index[target == self.classes[1]]  # index of points of second class
        d0 = data[index_0, :]  # data of points of first class
        t0 = target[index_0]
        d1 = data[index_1, :]  # data of points of second class
        t1 = target[index_1]
        # choose k points from each class
        index_choice_0 = np.random.choice(d0.shape[0], math.ceil((self.k / data.shape[0]) * d0.shape[0]), replace=False)
        index_choice_1 = np.random.choice(d1.shape[0], math.ceil((self.k / data.shape[0]) * d1.shape[0]), replace=False)
        # create data and target vectors to be used in the SVM
        A_data = np.empty([0, data.shape[1]])
        A_target = np.empty([0, target.shape[1]])

        A_data = np.append(A_data, d0[index_choice_0, :], axis=0)
        A_data = np.append(A_data, d1[index_choice_1, :], axis=0)
        A_target = np.append(A_target, t0[index_choice_0, :], axis=0)
        A_target = np.append(A_target, t1[index_choice_1, :], axis=0)
        return A_data, A_target

    def cost_function(self, data, target, w):
        # calculates the cost function of the model based on the data and weights
        loss = 0
        temp = target * np.dot(data, w.T)
        for i in range(data.shape[0]):
            loss += np.maximum(0, 1 - temp[i])  # compute hinge loss
        loss = loss / data.shape[0]
        return (loss + 0.5 * self.l * np.linalg.norm(w) ** 2)  # add regularization term


def myPegasos(filename, k, numruns):
    d = np.loadtxt(filename, delimiter=",")
    data = d[:, 1:]  # data
    values = d[:, 0]  # target values
    classes = np.unique(values)  # return the unique target classes
    target = separateClasses(classes, values)  # relabel target values at -1.0 and 1.0
    timer = []
    loss = {}
    for i in range(numruns):
        start = time.time()  # start times
        model = Pegasos(data, target, 1, k, 200000)  # run pegasos for 200,000 runs with lambda = 1
        loss[str(i)] = model.train(data, target)  # record loss
        finish = time.time()  # end timer
        timer.append(finish - start)
        print(timer[i])
    print("Mean Time: ", np.mean(timer))
    print("STD Time: ", np.std(timer))
    # plot the loss values for each run
    plt.figure()
    plt.title('Pegasos Cost for k = ' + str(k))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(range(len(loss['0'])), loss['0'])
    plt.plot(range(len(loss['1'])), loss['1'])
    plt.plot(range(len(loss['2'])), loss['2'])
    plt.plot(range(len(loss['3'])), loss['3'])
    plt.plot(range(len(loss['4'])), loss['4'])
    plt.yscale('log')
    plt.show()

    # return timer, loss