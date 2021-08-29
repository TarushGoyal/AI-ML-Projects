import numpy as np

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel

	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - dummy argment, don't use
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	return X @ Y.T
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf

	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - The sigma value for kernel
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	ans = np.zeros((X.shape[0],Y.shape[0]))
	ans -= np.sum(np.square(X),1).reshape((-1,1))
	ans -= np.sum(np.square(Y),1).reshape((1,-1))
	return np.exp((ans + 2 * X @ Y.T) / (2*sigma*sigma))
	# END TODO

def my_kernel(X,Y,sigma):
	'''Returns the gram matrix for your designed kernel

	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	a = 0.01
	return a * np.exp(linear_kernel(X[:,1].reshape((-1,1)),Y[:,1].reshape((-1,1)))) + \
		(linear_kernel(X[:,0].reshape((-1,1)),Y[:,0].reshape((-1,1)),0.1))**2
	# END TODO
