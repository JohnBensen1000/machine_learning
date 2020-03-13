from deepLearning import Neural_Network
import numpy as np # helps with the math
import statistics

num_hid = int(input('Number of hidden layers: '))
num_neu = int(input('Number of neurons per hidden layer: '))

print('\n')

num_in = int(input('Number of input data points: '))
max_in = int(input('Maximum possible input value: '))
num_out = int(input('Number of output data points: '))

Neural_Net = Neural_Network(num_in, num_hid, num_neu, num_out)
 
#####################################################################
    
def train_network(train_file, values_file):
    write_to = open(values_file, 'w') 
    raw_data = open(train_file, 'r')
    raw_data = raw_data.readlines()

    epoch = int(input('\nepoch: '))
        
    for i in range(1, epoch):
        in_list = raw_data[i].split(',')[1:num_in + 1]
        in_list = np.array([[float(a) / max_in] for a in in_list])         #turns data points into array of arrays--each insidea array is one float value from 0 to 1
        
        out_correct = [[0.0]] * num_out
        out_correct[int(raw_data[i].split(',')[0])] = [1.0]
                
        Neural_Net.feed_forward(in_list, out_correct)
        Neural_Net.back_prop(out_correct, in_list, i) 

        if((i % 1000) == 0): 
            print(str(i) + '           ' + str((Neural_Net.costFunc / i)) ) 

def find_accuracy(test_file):
    epoch = int(input('\nepoch: '))
    
    test_file = open(test_file, 'r')
    test_data = test_file.readlines()
     
    correct_count = 0

    for i in range(1, epoch):
        in_list = test_data[i].split(',')[1:num_in + 1]
        in_list = np.array([[float(a) / max_in] for a in in_list])         #turns data points into array of arrays--each insidea array is one float value from 0 to 1
        
        out_array = []
        out_correct = [0]  #This array wont be used, just a place holder
        
        Neural_Net.feed_forward(in_list, out_correct, findCorrect=False)
        if Neural_Net.outList.argmax() == int(test_data[i].split(',')[0]): correct_count += 1 
            
    print('------------------------------------')    
    print('accuracy = ' + str(correct_count / i))
    print('------------------------------------')  
    
def test_lines(test_file):
    test_file = open(test_file, 'r')
    test_data = test_file.readlines()
    
    test_line = ''
    
    while test_line != 'stop':
        test_line = int(input(" (Type 'stop' to stop testing) --- Test line: "))
        in_list = test_data[test_line].split(',')[1:num_in + 1]
        in_list = np.array([[float(a) / max_in] for a in in_list])         #turns data points into array of arrays--each insidea array is one float value from 0 to 1
        
        out_array = []
        out_correct = [0]  #This array wont be used, just a place holder
        
        Neural_Net.feed_forward(in_list, out_correct, findCorrect=False)
        out_array.append( Neural_Net.outList.argmax() )
        
        try:                
            print(statistics.mode(out_array))
        except:
            print("No result")
        
train_file  = input('Type train file name (should be excel file) ')
test_file   = input('Type test file name (should be excel file) ')

while 1 == 1:
    user_decide = input("Decide what to do ('train', 'find accuracy', 'test lines'): ")
    
    if(user_decide == 'train'): train_network(train_file, values_file)
        
    elif(user_decide == 'find accuracy'): find_accuracy(test_file)
    
    elif(user_decide == 'test lines'): test_lines(test_file)
    
    else: print("typo\n")
