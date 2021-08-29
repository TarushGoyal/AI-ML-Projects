import numpy as np
import argparse

def get_data(dataset):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train, Y_train, X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )

	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO 
	X = x.reshape((50,50))
	z = np.zeros((4,1))
	P1 = 0
	P2 = 0
	P3 = 0
	for i in range(50):
		for j in range(50):
			if i==0 or j==0 or i==49 or j==49:
				continue
			val = 4 - (X[i-1][j] + X[i+1][j] + X[i][j-1] + X[i][j+1])
			if val == 1:
				P1 += 1
			elif val == 2:
				P2 += 1
			elif val == 3:
				P3 += 1
	A = X.sum()
	z[3] = A/1000
	z[0] = (P1/100)**2
	z[1] = (P2/100)**2
	z[2] = (P3/100)**2
	return 	z.reshape((4,))
	### END TODO

class Perceptron():
    def __init__(self, C, D):
        '''
        C - number of classes
        D - number of features
        '''
        self.C = C
        self.weights = np.zeros((C, D))
        
    def pred(self, x):
        '''
        x - numpy array of shape (D,)
        '''
        ### TODO: Return predicted class for x
        val = np.dot(self.weights, x)
        return np.argmax(val)
        ### END TODO

    def train(self, X, Y, max_iter=10):
        for _ in range(max_iter):
            for i in range(X.shape[0]):
                ### TODO: Update weights
                y = self.pred(X[i,:])
                self.weights[Y[i]] += X[i,:]
                self.weights[y] -= X[i,:]
                ### END TODO

    def eval(self, X, Y):
        n_samples = X.shape[0]
        correct = 0
        for i in range(X.shape[0]):
            if self.pred(X[i]) == Y[i]:
                correct += 1
        return correct/n_samples

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data('D2')

	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')
