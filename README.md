# Machine Learning
A collection of machine learning algorithms

I personally am fascinated by machine learning and AI in general, and writing these programs was my way og gaining
a deeper understanding of how machine learning works. The following are brief descriptions of the three neural networks in 
this repository:

# deepLearning
  This is a deep neural network that trains through supervised learning. The deepLearning-control program asks the user
  for a .csv file for training, and then sends it to the deepLearning to train. Each line starts with an output, and the 
  rest of the line is the input that corresponds to the output. Through gradient descent and backpropagation, the network 
  learns to classify inputs with certain output. With a structure of 2 hidden neural networks and 32 neurons in each layer, 
  the network is able to achieve an accuracy rate of 94% on the MNIST dataset.
  
  This program has some user-adjustable hyperparameters. The user could decide how many hidden layers and how many 
  neurons there are per hidden layer. The user also inputs the .csv file for training, and a second .csv for testing.
  
  I relied on the video series produced by 3blue1brown a lot to understand how neural networks operate. In addition, I went to the websites
  towardsdatascience.com and machinelearningmastery.com a lot for additional help.
  
# auto_encoder
  This neural network works differently than the previous two networks. The point of this is to compress a set of input values,
  and then uncompress then to recreate the original image. The network has a one-dimensional hidden layer in between the input and 
  output layers. Through backpropagation and gradient descent, the network learns what the values for the hidden layer (with a 
  user-defined size) and weights and biases connecting the hidden layer and output layer should be for a certain input value. 
  
  The success of this network is determined by how similar the output layer is to the input layer. Ideally, they should be identical. 
 
# maxoutCNN
  This is a maxout convolutional neural network based on the paper Batch-normalized Maxout Network in Network, by Jia-Ren Chang and Yong-Sheng Chen.
  It is currently a work in progress, it contains one MIN block and 1 deep layer. At it's current state, it has an 86% accuracy on the MNIST dataset.
  More work is needed to scale the neural network and add more layers.
  
  The original paper could be found at https://arxiv.org/abs/1511.02583.
  
# Tensorflow
  I wrote this code along with a tutorial on how to user Tensorflow, found at https://www.youtube.com/watch?   v=tPYj3fFJGjk&list=PLYLyA78Q4izuAlaaOER3qwUZkqLTZf7qk&ab_channel=freeCodeCamp.org. The tutorial focused on deep
  neural networks, convolutional neural networks, data pre-processing, and transfer learning.
  
  

