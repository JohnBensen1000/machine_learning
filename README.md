# machine_learning
A collection of machine learning algorithms

I have uploaded three neural networks on this repository: auto_encoder, deepLearning, and CNN_Object. I have
also uploaded two other files, main-CNN and deepLearning-control, both of which are programs that allow the user
to interact with the CNN-Object and deepLearning programs, respectively. For the auto_encoder program, the code
that allows the user to interact with its neural network is in the same program.

I personally am fascinated by machine learning and AI in general, and writing these programs was my way og gaining
a deeper understanding of how machine learning works. The following are descriptions of the three neural networks in 
this repository:

deepLearning.py-
  This is a deep neural network that trains through supervised learning. The deepLearning-control program asks the user
  for a .csv file for training, and then sends it to the deepLearning to train. Each line starts with an output, and the 
  rest of the line is the input that corresponds to the output. Through gradient descent and backpropagation, the network 
  learns to classify inputs with certain output. 
  
  This program has some user-adjustable hyperparameters. The user could decide how many hidden layers and how many 
  neurons there are per hidden layer. The user also inputs the .csv file for training, and a second .csv for testing.
  
CNN_Object.py-
  Similarly to to the deepLearning.py, CNN_Object.py has a corresponding program, main-CNN.py that controls the CNN_Object.
  The main-CNN.py file asks the user for the training and testing .csv files. This program is configured for the MNIST data
  set. The main-CNN.py program takes the .csv data from the MNIST data set and creates a 28x28 image.
  
  This is a convolutionaly neural network. It looks at the image created by main-CNN.py program, creates a convoluted image 
  from this image, then creates a pool layer. Then, it creates a one-dimensional hidden layer with this data. This layer is
  connected to the output layer the same way two layers are connected in the deepLearning.py program. This network also learns
  to classify input values as a certain output value through gradient descent and backpropagation.
  
auto_encoder.py-
  This neural network works differently than the previous two networks. The point of this is to compress a set of input values,
  and then uncompress then to recreate the original image. The network has a one-dimensional hidden layer in between the input and 
  output layers. Through backpropagation and gradient descent, the network learns what the values for the hidden layer (with a 
  user-defined size) and weights and biases connecting the hidden layer and output layer should be for a certain input value. All 
  the weights/biases in this layer will be the same regardless of the input data. Therefore, when sending information from one 
  computer to another, the only thing that has to be sent is the hidden layer (which should have less neurons, and therefore less 
  information, than the input layer).
  
  The success of this network is determined by how similar the output layer is to the input layer. Ideally, they should be identical. 
  
I relied on the video series produced by 3blue1brown a lot to understand how neural networks operate. In addition, I went to the websites
towardsdatascience.com and machinelearningmastery.com a lot for additional help.
