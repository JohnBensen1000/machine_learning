from   CNN_Object2 import CNN
import numpy as np

train_file  = 'C:/Users/bense/Downloads/training_set.csv'
test_file   = 'C:/Users/bense/Downloads/mnist_test.csv'

epoch = int(input("Epoch of training set: "))
mainCNN = CNN(2, 16)

fileID    = open(train_file, 'r')
linesData = fileID.readlines()

for i in range(1, epoch):
    data_line = linesData[i].split(',')
    in_list = data_line[1:785]
    in_list = np.array([[float(a) / 255] for a in in_list])   

    outCorrect = [[0]] * 10
    outCorrect[int(data_line[0])] = [1.0]
    
    in_square = []

    for a in range(28):
        in_square.append(in_list[a*28: a*28 + 28])

    mainCNN.feedFor(in_square, outCorrect)      
    mainCNN.backProp(outCorrect, in_square)
    
    if i % 1000 == 0:    
        print(str(round(mainCNN.costFunc[0] / (i), 5)) + '     ' + str(i))

epoch = int(input("Epoch of test set: "))

numCorrect = 0
fileID    = open(test_file, 'r')
linesData = fileID.readlines()

for i in range(1, epoch):
    data_line = linesData[i].split(',')
    in_list = data_line[1:785]
    in_list = np.array([[float(a) / 255] for a in in_list])   

    outCorrect = [[0]] * 10
    outCorrect[int(data_line[0])] = [1.0]
    
    in_square = []

    for a in range(28):
        in_square.append(in_list[a*28: a*28 + 28])

    mainCNN.feedFor(in_square, outCorrect)      
    
    if np.argmax(mainCNN.outLayer) == int(data_line[0]):
        numCorrect += 1

print(numCorrect / epoch)

