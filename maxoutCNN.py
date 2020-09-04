#"C:/Users/bense/Downloads/mnist_test.csv"
#"C:/Users/bense/Downloads/training_set.csv"

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math, copy, random
from CNN import CNN

def graph_data_matrix(fileName, dataMatrix):
    plt.figure()
    plt.imshow(dataMatrix)
    plt.colorbar()
    plt.grid(False)
    plt.savefig("convLayers/" + fileName + ".png")
    
def print_data_matrix(dataMatrix):
    dataString = ""
    for r in range(len(dataMatrix)):
        for c in range(len(dataMatrix)):
            dataString += " " * (8 - len(str(round(dataMatrix[r, c], 5)))) + str(round(dataMatrix[r, c], 5)) + ", "

        dataString += "\n"
        
    testFile = open("testFile.txt", "w")
    testFile.write(dataString)

def make_two_dimensional(dataLine):
    ''' 
        Takes a line of data from a csv file, first data point is the label, the rest of the data points are 
        converted into a 28x28 numpy matrix, normalizes data points to be between -1 and 1
        Parameters:
            dataLine (string): a single line from a csv file
        Returns:
            dataLabel (int): the label for the row of data
            dataMatrix (numpy float matrix): a 28x28 matrix of the data points
            
    '''
    dataList   = dataLine.split(',')
    dataLabel  = np.zeros((10))
    dataMatrix = np.zeros((28, 28))
    
    for index, data in enumerate(dataList[1:]):
        dataMatrix[int(index / 28), index % 28] = (float(data) / 255) - .5
    dataLabel[int(dataList[0])] = 1

    return dataLabel, dataMatrix

def normalize(data):
    # takes a numpy matrix as an input, normalizes the elements in this matrix
    return (data - np.mean(data)) / (np.abs(np.std(data)) + 1)

def sigmoid(array, deriv=False):
        if deriv == False:
            return 1 / (1 + np.exp(-array))
        else:
            return array * (1 - array)
        
def create_conv(nextSize, filterWei, filterBias, prevLayer):
    # passes a filter over the previous layer to create the next convolutional
    # layer. Assumes that each layer is 28x28, and that the filter size is 3x3
    channel = np.zeros((nextSize, nextSize))

    for i in range(nextSize):
        for j in range(nextSize):
            section    = prevLayer[i:i + 3, j:j + 3].flatten()
            activation = np.dot(filterWei, section) + filterBias
            
            channel[i, j] = activation

    return normalize(channel)

class newCNN:
    def __init__(self, sizeHiddenLayer, numMaxOut):
        # build architecture of neural network, feature maps --> maxOut layer --> pooling layer --> deep network
        self.numMaxOut   = numMaxOut
        self.filterSize  = 3  
        self.initialSize = 28 
        
        # Creates architecture of convolutional portion of neural network.
        # Each successive layer decreases in both height and length by 2 units.
        self.convLayers, self.hiddenMax, self.maxOut = [], [], []
        for maxOut in range(self.numMaxOut):
            sizeDown = 6 * maxOut
            
            self.convLayers.append(np.zeros((3, 26 - sizeDown, 26 - sizeDown)))
            self.hiddenMax.append(np.zeros((9, 24 - sizeDown, 24 - sizeDown)))
            self.maxOut.append(np.zeros((3, 22 - sizeDown, 22 - sizeDown)))
        
        # counts how many filters there are in each convolutional layer, builds pooling layer
        # determines the sizes of each neural network layer
        self.numFilters  = (len(self.convLayers[0]) + len(self.hiddenMax[0]) + len(self.maxOut[0]))
        self.poolLayer   = np.zeros((3, (28 - 6 * self.numMaxOut) // 2, (28 - 6 * self.numMaxOut) // 2))
        self.deepLayers  = [3 * 11 ** 2, sizeHiddenLayer, 10]

        # fills in weights and biases with normally distributed random values
        self.filterWei, self.filterBia = [], []
        
        for i in range(self.numFilters * self.numMaxOut):
            self.filterWei.append( np.random.randn(self.filterSize ** 2) )
            self.filterBia.append( 0 )

        # Builds deep neural network architecture and inializes weights and biases 
        self.deepNetwork, self.weights, self.biases = [np.zeros(self.deepLayers[0])], [], []
        
        for layer in range(1, len(self.deepLayers)):
            self.deepNetwork.append( np.zeros(self.deepLayers[layer]) )
            self.weights.append(     np.random.randn(self.deepLayers[layer], self.deepLayers[layer - 1]) )
            self.biases.append(      np.zeros(self.deepLayers[layer]) )
        
        self.learningRate = .5
        
    def feed_forward(self, inputData):
        self.inputData = [inputData]
        
        for num in range(self.numMaxOut):
            self.build_conv_layer(inputData, num)
            self.create_max_pooling(num)
            
            if num < self.numMaxOut - 1:
                self.inputData.append(self.maxOut[num][0])

        self.create_pool_layer()
        self.feed_forward_deep_network()
        
    def build_conv_layer(self, inputData, num):     
        # Passes each filter across the input data to create the respective channels. Then
        # passes more filters over these channels to create the hidden maxout layer 
        
        for c in range(len(self.convLayers[num])):           
            n = c + (num * self.numFilters)
            self.convLayers[num][c] = create_conv(26, self.filterWei[n], self.filterBia[n], inputData)
        
        for c in range(len(self.hiddenMax[num])):
            n = c + len(self.convLayers[num]) + (num * self.numFilters)
            self.hiddenMax[num][c] = create_conv(24, self.filterWei[n], self.filterBia[n], self.convLayers[num][0])
    
    def create_max_pooling(self, num):   
        # The (i, j) pixel in the maxOut layer is the maximum value of 
        # corresponding (i, j) pixel from each channel.
        
        for c in range(len(self.maxOut[num])):
            for i in range(len(self.maxOut[num][0])):
                for j in range(len(self.maxOut[num][0])):
                    self.maxOut[num][0, i, j] = np.max( self.hiddenMax[num][3*c:3*c + 3, i, j] )  
        
        self.maxOut[num][0] = normalize(self.maxOut[num][0])
        
    def create_pool_layer(self):
        # Finds the maximum value of each block of four pixels to create a pool layer.     
        
        for c in range(len(self.poolLayer)):
            for i in range(len(self.poolLayer[0])):
                for j in range(len(self.poolLayer[0])):
                    self.poolLayer[c, i, j] = np.max(self.maxOut[-1][c, 2 * i:2 * i + 2, 2 * j:2 * j + 2])
            
            self.poolLayer[c] = normalize(self.poolLayer[c])
        
    def feed_forward_deep_network(self):
        # flattens the pool layer and continues with a deep neural network to find the results. The tanh 
        # function is used for all hidden layers, the sigmoid function is used for the output.
        
        self.deepNetwork[0] = self.poolLayer.flatten()

        for layer in range(1, len(self.deepLayers)):
            nextLayer = np.dot(self.weights[layer - 1], self.deepNetwork[layer - 1]) + self.biases[layer - 1]
            
            if layer < len(self.deepLayers) - 1:
                self.deepNetwork[layer] = np.tanh(nextLayer)
                
        self.deepNetwork[-1] = sigmoid(nextLayer)
                  
    def back_propogation(self, correctResults):
        resultError = self.back_propogate_deep_network(correctResults)
        resultError = np.resize(resultError, (3, 11, 11))

        for num in range(self.numMaxOut - 1, -1, -1):
            resultError = self.back_propogate_block(resultError, num)
        
    def back_propogate_deep_network(self, correctResults):
        # Performs backpropagation throught the deep network. Gradient descent is used to adjust all weights, biases, 
        # and activation levels. Returns the error that is passed back to the convolutional section of this neural network.
        resultError = correctResults - self.deepNetwork[-1]
        self.cost   = np.linalg.norm(resultError)

        for num in range(len(self.deepLayers) - 1, 0, -1): 
            changePrev   = np.zeros((self.deepLayers[num - 1]), dtype=float)
            currentLayer = self.deepNetwork[num]
            
            if num == len(self.deepLayers) - 1: activeError = sigmoid(currentLayer, deriv=True)
            else:                               activeError = 1 - currentLayer * currentLayer

            for row, delta in enumerate(np.multiply(resultError, activeError)):
                self.biases[num - 1][row] += delta * self.learningRate
                changePrev                += self.weights[num - 1][row, :] * delta

                for col, value in enumerate(self.deepNetwork[num - 1] * delta):
                    self.weights[num - 1][row, col] += value * self.learningRate

            resultError = changePrev

        return resultError
    
    def back_propogate_block(self, resultError, num):
        # Only backpropagates through the maximum activation in the set of activations that
        # fed into one pixel in the max pooling layer. tempChannels is matrix with the same
        # dimensions as the channels matrix, it temporarily holds the errors that are passed
        # back through backpropagation.
        tempChannels = np.zeros_like(self.hiddenMax[num])
        poolSize = 2 if num == self.numMaxOut - 1 else 1

        for k in range(len(resultError)):
            for i in range(len(resultError)):
                for j in range(len(resultError)):
                    crossChannel = self.hiddenMax[num][3*k:3*k + 3, poolSize * i:poolSize * i + poolSize, poolSize * j:poolSize * j + poolSize]
                    c, row, col  = np.where(crossChannel == np.max(crossChannel))
                    
                    tempChannels[c, poolSize * i + row, poolSize * j + col] = resultError[k, i, j]

        # Backpropogates through the maxout hidden layer. Keeps track of activation error
        # for the previous convolutional layer.
        prevError1 = np.zeros_like(self.convLayers[num])

        for c in range(len(self.hiddenMax[num])):
            n = len(self.convLayers[num]) + c + (self.numFilters * num)
            for i in range(24):
                for j in range(24):
                    if tempChannels[c, i, j] != 0:
                        section                      = self.convLayers[num][0, i:i + self.filterSize, j:j + self.filterSize].flatten()  
                        prevError1[0, i:i+3, j:j+3] += np.resize(self.filterWei[n], (3, 3)) * tempChannels[c, i, j]

                        self.filterWei[n] += section * tempChannels[c, i, j] * self.learningRate
                        self.filterBia[n] += tempChannels[c, i, j] * self.learningRate
        
        # backpropogates to the first convolutional layer 
        prevError2 = np.zeros_like(self.inputData[num])
                     
        for c in range(len(self.convLayers[num])):
            n = c + (self.numFilters * num)
            for i in range(26):
                for j in range(26):
                    if prevError1[c, i, j] != 0:
                        section                   = self.inputData[num][i:i + self.filterSize, j:j + self.filterSize].flatten()  
                        prevError2[i:i+3, j:j+3] += np.resize(self.filterWei[n], (3, 3)) * prevError1[0, i, j]
                        
                        self.filterWei[n] += section * prevError1[c, i, j] * self.learningRate
                        self.filterBia[n] += prevError1[c, i, j] * self.learningRate
                    
        return prevError2

def run_tests():
    testFile = open('C:/Users/bense/Downloads/mnist_test.csv', 'r')
    testData = testFile.readlines()
    
    correctCount = 0.0
    testCount    = 2000
    
    for row in range(testCount):
        _, dataMatrix = make_two_dimensional(trainData[row])
        correctAnswer = (int)(trainData[row].split(",")[0])

        cnn_1.feed_forward(dataMatrix)
        results         = cnn_1.deepNetwork[-1]
        answer, bestAns = -1, 0

        for index, value in enumerate(results):
            if value > bestAns:
                bestAns = value
                answer = index
                
        if answer == correctAnswer:
            correctCount += 1
        
    if correctCount > 0:
        print("Total cost: ", correctCount / testCount)

if __name__ == "__main__":               
    trainFile = open('C:/Users/bense/Downloads/training_set.csv', 'r')
    trainData = trainFile.readlines()
            
    costAve1, costAve2, costAve3 = [], [], []
    costList1, costList2, costList3 = [], [], []
    cnn_1 = newCNN(16, 1)
    maxSample = 50000

    for sample in range(1, maxSample):
        dataLabel, dataMatrix = make_two_dimensional(trainData[sample])
        
        cnn_1.feed_forward(dataMatrix)  
        cnn_1.back_propogation(dataLabel)
        costList1.append(cnn_1.cost)
        costAve1.append(sum(costList1) / len(costList1))
        '''
        cnn_3.feed_forward(dataMatrix)  
        cnn_3.back_propogation(dataLabel)
        costList3.append(cnn_3.cost)
        costAve3.append(sum(costList3) / len(costList3))
        '''
        if sample % 100 == 0:
            print(" " * (5 - len(str(sample))) + str(sample) + ": ", costAve1[-1])
            #print(sample)
    
    x = range(len(costAve1))
    plt.figure(1)
    plt.plot(x, costAve1)
    plt.title("Average cost over iterations")
    plt.savefig("figures/cost_over_time_2")

    run_tests()