'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form -
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer

        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary

    def forwardpass(self, X):
        '''

        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        if self.activation == 'relu':
            self.data = X @ self.weights + self.biases
            return relu_of_X(self.data)
        elif self.activation == 'softmax':
            self.data = X @ self.weights + self.biases
            return softmax_of_X(self.data)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  :
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO
        if self.activation == 'relu':
            pred = self.data
            delta1 = gradient_relu_of_X(pred, delta)
            delta2 = delta1 @ self.weights.T
            self.weights -= lr * (activation_prev.T @ delta1)
            self.biases -= lr * np.sum(delta1,0)
            return delta2
        elif self.activation == 'softmax':
            pred = self.data
            delta1 = gradient_softmax_of_X(pred, delta)
            delta2 = delta1 @ self.weights.T
            self.weights -= lr * (activation_prev.T @ delta1)
            self.biases -= lr * np.sum(delta1,0)
            return delta2
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))
        self.biases = np.random.normal(0,0.1,self.out_depth)


    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        s = self.stride
        if self.activation == 'relu':
            result = np.zeros((X.shape[0],self.out_depth,self.out_row,self.out_col))
            for j in range(self.out_row):
                for k in range(self.out_col):
                    result[:,:,j,k] = np.sum(
                        self.weights[np.newaxis,:,:,:,:] *
                        X[:,np.newaxis,:, s*j : s*j+self.filter_row, s*k : s*k+self.filter_col],
                        (2,3,4)
                    )
            result += self.biases[np.newaxis,:,np.newaxis,np.newaxis]
            self.data = result
            # print(relu_of_X(result))  
            return relu_of_X(result)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev

        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        if self.activation == 'relu':
            delta1 = gradient_relu_of_X(self.data, delta)
            s = self.stride
            n = delta1.shape[0]
            #
            db = delta1.sum(axis=(0, 2, 3))
            dw = np.zeros(self.weights.shape)
            result = np.zeros(activation_prev.shape)
            for i in range(self.out_row):
                for j in range(self.out_col):
                    row_start = i * s
                    row_end = i * s + self.filter_row
                    col_start = j * s
                    col_end = j * s + self.filter_col
                    result[:, :, row_start:row_end, col_start:col_end] += np.sum(
                        # batch_size, out_depth, in_depth, filter_row, filter_col
                        self.weights [np.newaxis, :, :, :, :] *
                        delta1[:,:,np.newaxis, i:i+1, j:j+1],
                        axis = 1
                    )
                    dw += np.sum(
                        activation_prev[:,np.newaxis,:,row_start:row_end,col_start:col_end] *
                        delta1[:, :, np.newaxis, i:i+1, j:j+1],
                        axis = 0
                    )
            self.weights -= lr * dw
            self.biases -= lr * db
            return result
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO

class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer

        # TODO
        s = self.stride
        result = np.zeros((X.shape[0],self.out_depth,self.out_row,self.out_col))
        for i in range(self.out_row):
            for j in range(self.out_col):
                result[:,:,i,j] = np.sum(
                    X[:,:, s*i : s*i+self.filter_row, s*j : s*j+self.filter_col],
                    (2,3)
                )/(self.filter_row * self.filter_col)
        return result
        # END TODO
        ###############################################

    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev

        # TODO
        s = self.stride
        n = delta.shape[0]
        result = np.zeros(activation_prev.shape)
        for i in range(self.out_row):
            for j in range(self.out_col):
                row_start = i * s
                row_end = i * s + self.filter_row
                col_start = j * s
                col_end = j * s + self.filter_col
                result[:, :, row_start:row_end, col_start:col_end] += (delta[:,:,i:i+1, j:j+1]/(self.filter_row * self.filter_col))
        return result
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer

        # TODO
        s = self.stride
        result = np.zeros((X.shape[0],self.out_depth,self.out_row,self.out_col))
        for i in range(self.out_row):
            for j in range(self.out_col):
                result[:,:,i,j] = np.max(
                    X[:,:, s*i : s*i+self.filter_row, s*j : s*j+self.filter_col],
                    axis = (2,3)
                )
        return result
        # END TODO
        ###############################################

    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev

        # TODO
        s = self.stride
        n = delta.shape[0]
        result = np.zeros(activation_prev.shape)
        for i in range(self.out_row):
            for j in range(self.out_col):
                row_start = i * s
                row_end = i * s + self.filter_row
                col_start = j * s
                col_end = j * s + self.filter_col
                tmp = np.max(
                    activation_prev[:, :, row_start:row_end, col_start:col_end], (2,3)
                )[:,:,np.newaxis,np.newaxis]
                mask = activation_prev[:, :, row_start:row_end, col_start:col_end] == tmp
                mask_sum = np.sum(mask,(2,3))[:,:,np.newaxis,np.newaxis]
                mask = mask / mask_sum
                result[:, :, row_start:row_end, col_start:col_end] += delta[:,:,i:i+1, j:j+1] * mask
        return result
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass

    def forwardpass(self, X):
        # TODO
        return X.reshape((X.shape[0],-1))
    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(activation_prev.shape)
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    return (X > 0) * X
    # END TODO

def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass

    # TODO
    return delta * (X > 0)
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax

    # TODO
    tmp = np.exp(X)
    return tmp / np.sum(tmp,1).reshape((-1,1))
    # END TODO
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    p = softmax_of_X(X)
    o = X.shape[1]
    result = np.zeros((X.shape[0],o))
    for b in range(X.shape[0]):
        p_ = p[b,:].reshape(o,1)
        aneesh = p[b,:] * np.eye(o) - p_ @ p_.T
        result[b,:] = delta[b,:].reshape((1,o)) @ aneesh
    return result
    # END TODO
