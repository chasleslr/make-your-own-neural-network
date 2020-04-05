import numpy as np
import scipy.special


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate, init_weight='normal'):

        # network parameters
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lrate = learning_rate

        if init_weight == 'uniform':
            # initial weights (uniform distribution)
            self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
            self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        if init_weight == 'normal':
            # initial weights (normal distribution [1 / sqrt(incoming links])
            self.wih = np.random.normal(0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # activation function (sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):

        # convert inputs and targets to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # compute signals in/out the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # compute signals in/out the output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # compute the errors at each layer
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights between each layer
        self.who += self.lrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):

        # convert inputs to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # compute signals in/out the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # compute signals in/out the output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def export_training(self):

        return self.who, self.wih

    def load_training(self, who, wih):

        self.who = who
        self.wih = wih
        
        pass
