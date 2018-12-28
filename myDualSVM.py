def separateClasses(classes,values):
    # this function classifies data corresponding to 1 as -1.0 
    # and target values corresponding to 3 as 1.0
    
    target = np.zeros([values.shape[0],1])
    for i in range(0,target.shape[0]):
        if values[i] == classes[0]:
            target[i] = -1.0
        else:
            target[i] = 1.0
    return target

# import libraries
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import math
import matplotlib.pyplot as plt

class SVM(object):
    def __init__(self,data,target,c):
        self.C = c # the C parameter
        self.classes = np.unique(target) # different target classes
        self.features = data.shape[1] # number of total features
        self.samples = data.shape[0] # number of data points for the training data
        
    def predictor(self,data,target):
    # This function takes in data and reutrns the classification error
    # for both the training and testing data. The prediction is calculated
    # using y = sign(data * weight + bias)
        prediction = np.zeros(target.shape)
        for i in range(0,data.shape[0]):
            prediction[i] = np.sign(np.dot(data[i,:],self.w)+self.b)
            # use the sign function to classify predictions as -1.0 or 1.0
        error = np.sum(target != prediction)/prediction.shape[0] # calculare the error
        return error
    
    def fit(self, data, target,test_data,test_target):
    # This function runs the fits the Dual Form SVM on the data using 
    # CVXOPT to compute alpha values of the SVM and uses alpha values
    # greater than 1e-4 (threshold) to determine support vectors, bias
    # terms and weights. The computed weights and bias were used to 
    # compute the training error and testing error. 
   
        K = target*data # the kernel function
        P = matrix(np.dot(K,K.T))
        q = matrix(-np.ones([self.samples,1]))
        G = matrix(np.vstack((np.identity(self.samples),-1*np.identity(self.samples))))
        h = matrix(np.vstack((np.ones((self.samples,1))*self.C,np.zeros((self.samples,1)))))
        A = matrix(target.reshape(1,-1))
        b = matrix(np.zeros(1))
        
        solution = solvers.qp(P,q,G,h,A,b)
        alpha = np.array(solution['x'])
        sv = alpha > 1e-4
        self.alpha = alpha[sv].reshape([np.sum(sv),1])
        self.n_sv = np.sum(sv)
        index = np.where(sv)
        self.support_vector = data[index[0],:]
        self.support_vector_y = target[sv].reshape([np.sum(sv),1])
        self.w = np.sum(self.alpha*self.support_vector_y*self.support_vector,axis=0)
        self.margin = 1/np.linalg.norm(self.w)
        temp = np.dot(data,self.w).reshape(target.shape)
        self.b = -(np.min(temp[target==1.]) + np.max(temp[target==-1.]))/2
        train_error = self.predictor(data,target)
        test_error = self.predictor(test_data,test_target)
        return test_error, train_error



def myDuaSVM(filename, C):
    # This function runs the Dual Form SVM on the data 
    # load in the data from csv
    d = np.loadtxt(filename,delimiter=",")
    data = d[:,1:] # data
    data = data/np.linalg.norm(data,axis=1,keepdims=True) # normalize the data
    values = d[:,0] # target values
    classes = np.unique(values) # return the unique target classes
    target = separateClasses(classes,values) # relabel target values at -1.0 and 1.0
    n_folds = 10 # to run the cross fold validation
    
    # create empty arrays to store data
    test_error = np.zeros([n_folds,len(C)])
    train_error = np.zeros([n_folds,len(C)])
    margin = np.zeros([n_folds,len(C)])
    n_sv = np.zeros([n_folds,len(C)])
    
    index = np.arange(data.shape[0])
    solvers.options['show_progress'] = False # prevents solver from printing
    for k in range(0,n_folds):
        test_size = math.floor(data.shape[0]*0.2) # size of test data
        test_index = np.random.choice(d.shape[0],test_size,replace=False) # test data index
        test_data = data[test_index,:] # test data
        test_target = target[test_index]
        train_index = [x for x in index if x not in test_index] # training data index
        train_data = data[train_index,:] # training data
        train_target = target[train_index]
        for i in range(0,len(C)):
            # create SVM model with training data
            model = SVM(train_data,train_target,C[i])
            # fit the model and return the testing error and training error
            test_error[k,i], train_error[k,i] = model.fit(train_data,train_target,test_data, test_target)
            n_sv[k,i] = model.n_sv
            margin[k,i] = model.margin
    # create a dictionary of the data
    dic = {'# Support Vectors (AVG)':np.mean(n_sv,axis=0),'# Support Vectors (STD)': np.std(n_sv,axis=0),
          'Training Error (AVG)':np.mean(train_error,axis=0),'Training Error (STD)': np.std(train_error,axis=0),
          'Testing Error (AVG)': np.mean(test_error,axis=0),'Testing Error (STD)': np.std(test_error,axis=0),
          'Margin (AVG)':np.mean(margin,axis=0),'Margin (STD)': np.std(margin,axis=0)}
    print(pd.DataFrame.from_dict(dic, orient='index',columns = ['C = 0.01','C = 0.1','C = 1','C = 10','C = 100']))
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.bar([str(x) for x in C], dic['Testing Error (AVG)'], xerr=0, yerr=dic['Testing Error (STD)'])
    plt.xlabel('C Value')
    plt.ylabel('Test Error')
    plt.title('Test Error vs C-Value')

    plt.subplot(2,2,2)
    plt.bar([str(x) for x in C], dic['Training Error (AVG)'], xerr=0, yerr=dic['Training Error (STD)'])
    plt.xlabel('C Value')
    plt.ylabel('Training Error')
    plt.title('Training Error vs C-Value')

    plt.subplot(2,2,3)
    plt.bar([str(x) for x in C], dic['# Support Vectors (AVG)'], xerr=0, yerr=dic['# Support Vectors (STD)'])
    plt.xlabel('C Value')
    plt.ylabel('# Support Vectors')
    plt.title('# Support Vectors vs C-Value')

    plt.subplot(2,2,4)
    plt.bar([str(x) for x in C], dic['Margin (AVG)'], xerr=0, yerr=dic['Margin (STD)'])
    plt.xlabel('C Value')
    plt.ylabel('Margin Size')
    plt.title('Margin Size vs C-Value')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    
    
    #return dic