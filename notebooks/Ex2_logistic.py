import matplotlib.pyplot as plt
from Logistic_Regression import *
import pdb

def bp():
    pdb.set_trace()

def plotData(X, y):
    pos_idx = [idx for (idx, val) in enumerate(y[:]) if (val == 1)]
    neg_idx = [idx for (idx, val) in enumerate(y[:]) if (val == 0)]
    plt.scatter(X[pos_idx, 0], X[pos_idx, 1], c = 'r', marker = '+') 
    plt.scatter(X[neg_idx, 0], X[neg_idx, 1], c = 'y', marker = 'o')

if __name__ == '__main__':
    data = np.loadtxt('data/ex2data1.txt', delimiter = ',')
    print 'data shape: ', data.shape

    X = data[:, 0:2]
    y = data[:, 2]

    m = y.shape[0]
    num_Features = X.shape[1] if (len(X.shape) != 1) else 1
    num_Labels = y.shape[1] if (len(y.shape) != 1) else 1

    #########################################
    ##  1. Plotting
    ##  2017.11.02
    #########################################
    plt.figure('Ex2')
    plotData(X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'], loc = 'best')
    #plt.show()


    #########################################
    ##  2. Compute Cost and Gradient
    ##  2017.11.02
    #########################################
    X = np.c_[np.ones([m, 1]), X]
    y = y.reshape(-1, num_Labels)

    initial_theta = np.zeros([num_Features + 1, 1])

    cost, grad = costFunction(initial_theta, X, y)
    print 'Cost at initial theta (zeros): %f' % cost
    print 'Expected cost (approx): 0.693'
    print 'Gradient at initial theta (zeros):'
    print grad
    print 'Expected gradients (approx):'
    print ' -0.1000\n -12.0092\n -11.2628'

    test_theta = np.c_[[-24, 0.2, 0.2]]
    cost, grad = costFunction(test_theta, X, y)
    print 'Cost at initial theta (zeros): %f' % cost
    print 'Expected cost (approx): 0.218'
    print 'Gradient at initial theta (zeros):'
    print grad
    print 'Expected gradients (approx):'
    print ' 0.043\n 2.566\n 2.647'


    #########################################
    ##  3. 
    ##  2017.11.03
    #########################################
