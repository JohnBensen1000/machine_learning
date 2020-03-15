#This is a Neural Network. The design was based heavily on the video series, by 3blue1brown, that explained how a neural network works. 
#There were several articals on towardsdatascience.com that were also very useful
#The program asks the operator for the number of hidden layers and number of neurons in each hidden layer
#The program also asks for the number of neural networks to train. To increase accuracy, multiple neural networks are trained, when
#predicting the answer, the results from these networks are compared. Whichever answer shows up the most is recored as being the predicted answer
#all input values are converted from a value from 0 to 1, all values are divided by max value
#This also uses the sigmoid function as the activation function

#training data has to be in the following format:
    # 5,0,0,0,1,24,255,...,45,23,0,0,0
    # First number represents the correct answer, the following numbers represent the data points
    # Should be excel file
    
import random
import numpy as np # helps with the math
import statistics
from numpy import dot, exp, multiply, add

class Neural_Network:
    def __init__(self, numIn, numHid, numNeu, numOut):
        #number of activation layers = number of hidden plus output layer
        self.numLayers = numHid + 1                                                             
        self.numOut    = numOut
        #initializes random weights/biases between input layer and first hidden layer
        self.wList, self.bList = [np.random.randn(numNeu, numIn)], [np.zeros((numNeu, 1))]    

        for i in range(numHid - 1):                                    
            self.wList.append(np.random.randn(numNeu, numNeu))   
            self.bList.append(np.zeros((numNeu, 1)))

        self.wList.append(np.random.randn(numOut, numNeu))           
        self.bList.append(np.zeros((numOut, 1)))

        self.learnRate = 1.1
        self.costFunc  = 0
        
    def activation(self, z, deriv=False):         
         #activation function ==> S(x) = 1/1+e^(-x), derivative of activation function ==> z * (1 - z)
        if deriv == True:
            return z * (1 - z)
        return 1 / (1 + exp(-z))
        
    def feed_forward(self, inList, outCorrect, findCorrect=True):                                   
        #finds activations for each hidden layer and output layer
        self.aList = [self.activation(dot( self.wList[0], inList ) + self.bList[0], False)]  
        #goes through each hidden layer plus the final layer     
        for a in range(1, self.numLayers):
            self.aList.append(self.activation(dot( self.wList[a], self.aList[a - 1] ) + self.bList[a]))    
        #output layer set to last layer in aList          
        self.outList = self.aList[a]    

        if(findCorrect == True):
            for i in range(self.numOut): self.costFunc += (((self.outList[i] - outCorrect[i]) ** 2) / self.numOut)   
              
    def back_prop_layer(self, a, tempA0, tempDA1, adjustPre=True ):
        dz = self.activation(self.aList[a], deriv=True)
        #adjusts format of a2_list so it could be used          
        tempA0 = tempA0.flatten()                                                       
        
        tempDA0 = np.array( [0.0] * len(tempA0) )                                       
        if(adjustPre == True): 
            #finds total adjustments that should be made for previous activation, does not perform this task for input layer
            for i in range(len(self.aList[a])): 
                tempDA0 = add(tempDA0, (tempDA1[i] * dz[i]) * self.wList[a][i])      

        self.wList[a] -= multiply([(i * tempA0) for i in (tempDA1.flatten() * dz.flatten())], self.learnRate)     
        self.bList[a] -= multiply([[i] for i in (tempDA1.flatten() * dz.flatten())]         , self.learnRate)
        
        return multiply(tempDA0, self.learnRate)              
            
    def back_prop(self, outCorrect, inList, count): 
        #starts at output layer and works backwards, adjusts all weights/biases for single test case
        daList = [(self.outList - outCorrect).flatten()]
        
        for a in range(self.numLayers - 1, 0, -1): 
            #da = a - a_correct, change in previos activation should be the difference between what the activation is and what it should be,       
            daList.append( self.back_prop_layer(a, self.aList[a - 1], daList[self.numLayers - 1 - a]) )
        self.back_prop_layer(0, inList, daList[self.numLayers - 1 - 0], adjustPre=False)   
        
        learnCount = (count % 5000) / 5000
        self.learnRate = 1.1 - learnCount
        
