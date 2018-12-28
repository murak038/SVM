# SVM
As an assignment for CSCI 5525 - Machine Learning, we had to implement a linear SVM in dual form, the Pegasos algorithm described in “Pegasos: Primal Estimated sub-GrAdient SOlver for SVM” and the softplus gradient descent algorithm on the MNIST-13 dataset. 

### Requirements to Run Scripts:
`python 3.6`
`numpy 1.15.0`
`matplotlib 2.2.3`
`pandas 0.23.4`
`cvxopt`
`json`
`math`
`time`

### Functions: 
1. myDualSVM(filename, C)
2. myPegasos(filname,k,numruns)
3. mySoftplus(filename,k,numruns)


### Problem 1: `myDualSVM(filename, C)`

Goal: To implement a linear SVM with slack variables in dual form using the CVXOPT library to determine support vectors and associated weights. 

Input: 
- filename: name and path of csv file containing target in first column and data in the other
- C: the penalty placed on the slack variables.

Outputs: returns a table containing the mean number of support vectors, mean training error, mean testing error and mean margin size for each value of C and bar charts for each.

Instructions to Run in Terminal:
```
from myDualSVM import *
myDualSVM(filename,C)
```


### Problem 2a: `myPegasos(filename, k,numruns)`

Goal: Implement the Pegasos algorithm to evaluate its performance on the MNIST-13 dataset for varying batch sizes.

Input:
- filename: name and path of csv file containing target in first column and data in the other
- k: mini-batch size for Pegasos ( a singular number)
- numruns: the number of runs 

Output: returns the mean and standard deviation of the run time for each run and a plot of the loss function for each run.

Instructions to Run in Terminal:
```
from myPegasos import *
myPegasos(filename,k,numruns)
```


### Problem 2b: `mySoftplus(filename, k,numruns)`

Goal: Implement the Pegasos algorithm to evaluate its performance on the MNIST-13 dataset for varying batch sizes.

Input:
- filename: name and path of csv file containing target in first column and data in the other
- k: mini-batch size  ( an array of all values)
- numruns: the number of runs 

Output: returns the training loss of each run, plot of the training loss and plot of runtimes for each value of k.

Instructions to Run in Terminal:
```
from mySoftplus import *
mySoftplus(filename, k, numruns)
 ```

Information About Datasets:
MNIST-13: 2000 points, 784 features, 2 target variables



