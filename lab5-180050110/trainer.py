'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 10
			self.lr = 2e-4
			self.nn = nn.NeuralNetwork(10, self.lr)
			self.nn.addLayer(FullyConnectedLayer(784,256,"relu"))
			self.nn.addLayer(FullyConnectedLayer(256,10,"softmax"))


		if dataset_name == 'CIFAR10':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]

			self.save_model = True
			self.model_name = "model.p"

			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 30
			self.lr = 7e-4
			self.nn = nn.NeuralNetwork(10, self.lr)
			self.nn.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 4, 1, 'relu'))
			self.nn.addLayer(AvgPoolingLayer([4, 30, 30], [2, 2], 2))
			self.nn.addLayer(ConvolutionLayer([4, 15, 15], [4, 4], 4, 1, 'relu'))
			self.nn.addLayer(MaxPoolingLayer([4, 12, 12], [2, 2], 2))
			self.nn.addLayer(FlattenLayer())
			self.nn.addLayer(FullyConnectedLayer(144, 10, 'softmax'))

		if dataset_name == 'XOR':
			## works for all seeds except default lol!
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 100
			self.epochs = 200
			self.lr = 0.0001
			self.nn = nn.NeuralNetwork(2, self.lr)
			self.nn.addLayer(FullyConnectedLayer(2,5,"relu"))
			self.nn.addLayer(FullyConnectedLayer(5,2,"softmax"))


		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 200
			self.epochs = 200
			self.lr = 1e-4
			self.nn = nn.NeuralNetwork(2, self.lr)
			self.nn.addLayer(FullyConnectedLayer(2,2,"relu"))
			self.nn.addLayer(FullyConnectedLayer(2,2,"softmax"))
	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net

		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# A Training Epoch
			# if verbose:
			# 	print("Epoch: ", epoch)

			# TODO
			# Shuffle the training data for the current epoch
			index = np.arange(self.XTrain.shape[0])
			np.random.shuffle(index)
			self.XTrain = self.XTrain[index,:]
			self.YTrain = self.YTrain[index,:]
			# Divide the training data into mini-batches

				# Calculate the activations after the feedforward pass
				# Compute the loss
				# Calculate the training accuracy for the current batch
				# Backpropagation Pass to adjust weights and biases of the neural network
			# for i in range(30):
			# 	print(self.XTrain[i,:] , self.YTrain[i,:])
			numBatches = self.XTrain.shape[0] // self.batch_size
			for batch_id in range(numBatches):
				X = self.XTrain[batch_id * self.batch_size : batch_id * self.batch_size + self.batch_size , :]
				Y = self.YTrain[batch_id * self.batch_size : batch_id * self.batch_size + self.batch_size , :]
				activations = self.nn.feedforward(X)
				# trainAcc += acc
				self.nn.backpropagate(activations, Y)
			activations = self.nn.feedforward(self.XTrain)
			pred, trainAcc = self.nn.validate(self.XTrain,self.YTrain)
			trainLoss = self.nn.computeLoss(self.YTrain, activations)
			# END TODO
			# Print Training loss and accuracy statistics
			# trainAcc /= numBatches
			if verbose:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)

			if self.save_model:
				model = []
				for l in self.nn.layers:
					# print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer" and type(l).__name__ != "MaxPoolingLayer":
						model.append(l.weights)
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and verbose:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)
