import numpy as np
import argparse
from utils import *


class BinaryLogisticRegression:
    def __init__(self, D):
        """
        D - number of features
        """
        self.D = D
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        X - numpy array of shape (N, D)
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        tmp = 1/(1 + np.exp(-X @ self.weights))
        return tmp > 0.5
        # END TODO

    def train(self, X, Y, lr = 2.5, max_iter = 50):
        for _ in range(max_iter):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            preds = 1/(1 + np.exp(-X @ self.weights))
            grad  = (np.sum((Y - preds) * X , axis = 0).reshape((-1,1)))/X.shape[0]
            self.weights = self.weights + lr * grad     
            # END TODO

            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            # print(_,np.linalg.norm(grad))
            if np.linalg.norm(grad) <= 1e-4:
                break
            # End TODO

    def accuracy(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        accuracy = ((preds == Y).sum()) / len(preds)
        return accuracy

    def f1_score(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        # TODO: calculate F1 score for predictions preds and true labels Y
        precision = sum(preds + Y == 2)/sum(preds == 1)
        recall = sum(preds + Y == 2)/sum(Y == 1)
        return 2 * precision * recall / (precision + recall)
        # End TODO


if __name__ == '__main__':
    np.random.seed(335)

    X, Y = load_data('data/songs.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    D = X_train.shape[1]
    lr = BinaryLogisticRegression(D)
    lr.train(X_train, Y_train)
    preds = lr.predict(X_test)
    acc = lr.accuracy(preds, Y_test)
    f1 = lr.f1_score(preds, Y_test)
    print(f'Test Accuracy: {acc}')
    print(f'Test F1 Score: {f1}')
