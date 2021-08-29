import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)

def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    ## TODO
    N = X.shape[0]
    diff = X@W - Y 
    mse = (diff.T @ diff)/(2*N)
    ## END TODO

    return mse[0][0]


def ista(X_train, Y_train, X_test, Y_test, _lambda = 0.1, lr = 0.004, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    N,nf = X_train.shape
    W = np.random.randn(nf,1)
    # END TODO
    tail = X_train.T @ Y_train
    head = X_train.T @ X_train
    for i in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        # END TODO
        train_mses.append(train_mse)
        test_mses.append(test_mse)
        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.
        grad = (head@W - tail)/N
        new_W = W - lr * grad
        new_W[new_W > _lambda * lr]-= _lambda * lr
        new_W[new_W < -_lambda * lr]+= _lambda * lr
        # END TODO
        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if np.linalg.norm(new_W - W) < 1e-4:
            break
        else:
            W = new_W;
        # End TODO

    return W, train_mses, test_mses

def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr = 0.01, max_iter = 2000):
    '''
    reg - regularization parameter (lambda in Q2.1 c)
    '''
    train_mses = []
    test_mses = []

    ## TODO: Initialize W using using random normal 
    N,nf = X_train.shape
    W = np.random.randn(nf,1)

    tail = X_train.T @ Y_train
    head = X_train.T @ X_train
    ## END TODO
    for i in range(max_iter):

        ## TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        ## END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        ## TODO: Update w and b using a single step of gradient descent
        grad = (head@W - tail)/N + 2*reg*W
        W -= lr * grad
        ## END TODO
    return W, train_mses, test_mses

def scatter_plot(X_train,Y_train,X_test,Y_test,_lambda):
    W_lasso, _, _ = ista(X_train, Y_train, X_test, Y_test,_lambda,0.001,30000)
    W_ridge, _, _ = ridge_regression(X_train, Y_train, X_test, Y_test,10,0.001,30000)
    fig, axs = plt.subplots(2)
    fig.suptitle('Ridge and Lasso')
    axs[1].scatter([_ for _ in range(W_lasso.shape[0])], W_lasso, alpha = 0.2)
    axs[1].set_xlabel('Index')
    # print(_lambda)
    axs[1].set_ylabel('Lasso with lambda = %.2f'%_lambda)
    axs[0].scatter([_ for _ in range(W_ridge.shape[0])], W_ridge, alpha = 0.2)
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Ridge with lambda = 10')

def lambda_plot(X_train, Y_train, X_test, Y_test):
    test_mses = []
    train_mses = []
    _lambdas = [0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.25,0.3,0.5,1,2,3,4,5,6]
    best_lambda = -1
    best_mse = 1e6
    for _lambda in _lambdas:
        W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test,_lambda,0.001,10000)
        test_mses.append(test_mses_ista[-1])    
        train_mses.append(train_mses_ista[-1])
        if best_mse > test_mses[-1]:
            best_mse = test_mses[-1]
            best_lambda = _lambda
    plt.figure(0)
    plt.plot(_lambdas,train_mses)
    plt.plot(_lambdas,test_mses)
    plt.legend(['train','test'])
    plt.xlabel('lambda')
    plt.ylabel('mse')
    return best_lambda

if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    best_lambda = lambda_plot(X_train,Y_train,X_test,Y_test)
    scatter_plot(X_train,Y_train,X_test,Y_test,best_lambda)
    plt.show()
    # End TODO
