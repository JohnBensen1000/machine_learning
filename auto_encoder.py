import random
import numpy as np # helps with the math
import statistics
import cProfile
import re

from numpy import dot, subtract, zeros, power, random, exp, array, absolute, multiply

raw_file = open('C:/Users/bense/Downloads/training_set.csv', 'r')
raw_data = raw_file.readlines()

learning_rate = float(input('learning rate: '))
batch_size = int(input('Batch size: '))
num_a = int(input('Number of neurons in bottleneck layer: '))

class Encoder:
    def __init__(self, num_a, num_in, learning_rate, batch_size):
        self.num_a, self.num_in = num_a, num_in
        self.le_ra, self.ba_si  = learning_rate, batch_size
        
        self.w_list = [random.rand(num_a, num_in) - .5, random.rand(num_in, num_a) - .5]
        self.b_list = [zeros((num_a, 1)), zeros((num_in, 1))]
        
        self.w_temp = [zeros((num_a, num_in)), zeros((num_in, num_a))]
        self.b_temp = [zeros((num_a, 1)), zeros((num_in, 1))]
        
        self.cost = 0.0

    def activation(self, z, sigmoid, deriv=False):    
        if sigmoid == True:
            if deriv == True:
                return z * (1 - z)
            return 1 / (1 + exp( multiply(z, -1) ))
        
        else:
            if deriv == True:
                return (absolute(z) + z) / (2 * z + .0001)
            return (absolute(z) + z) / 2
        
    def back_prop(self, la, da_1, cur_list, pre_list, sigmoid=True, pre_adjust=True):
        dz = self.activation(cur_list, sigmoid, deriv=True).flatten()
        pre_list = pre_list.flatten()
        
        const = self.le_ra / self.ba_si
        
        self.w_temp[la] += multiply([pre_list * a for a in (da_1 * dz)], const)
        self.b_temp[la] += multiply([[a] for a in (da_1 * dz)], const)
        
        if pre_adjust == True:
            da_0 = array( [0.0] * len(pre_list) )
            for a in range(len(da_1)): 
                if(abs(da_1[a] * dz[a]) > .25): da_0 += da_1[a] * dz[a] * self.w_list[la][a] 
                
            return multiply(da_0, self.le_ra)        
    
    def update_values(self):
        self.w_list, self.b_list = subtract(self.w_list, self.w_temp), subtract(self.b_list, self.b_temp)
        self.w_temp = [zeros((self.num_a, self.num_in)), zeros((self.num_in, self.num_a))]
        self.b_temp = [zeros((self.num_a, 1)), zeros((self.num_in, 1))]
        
    def zip_data(self, in_list, i, unzip=False, train=False):  
        self.a_hid = self.activation( dot( self.w_list[0], in_list ) + self.b_list[0], False )
        
        if unzip == True:
            self.a_out =  self.activation( dot( self.w_list[1], self.a_hid ) + self.b_list[1], True) 
            for a in (self.a_out - in_list): self.cost += power(a, 2) / 784
        
        if train == True: 
            da_1 = (self.a_out - in_list).flatten()
            da_0 = self.back_prop(1, da_1, self.a_out, self.a_hid)
            self.back_prop(0, da_0, self.a_hid, in_list, False, False)
            
            if i % self.ba_si == 0: self.update_values()
                
Enc_1 = Encoder(num_a, 784, learning_rate, batch_size)

def train_zip(raw_data, epoch):
    count = 0
    
    for i in range(epoch):
        count += 1

        in_list = array([[float(a) / 255] for a in raw_data[i].split(',')[1:785]])
        Enc_1.zip_data(in_list, i, True, True)
        if( count % 1000 == 0 ): print(str(count) + '        ' + str(Enc_1.cost / count))
        
#cProfile.run('train_zip(raw_data)')

def draw_image(data):
    out_string, c = '', 0
    
    for a in data:
        if a < .1: out_string += ' ' * 6
        else: out_string += str( round(a[0], 2) ) +  ' ' * (5 - len(str(round(a[0], 2))) )
        
        if c % 28 == 0: out_string += '\n'
        c += 1
        
    print(out_string)
    
stop = ''

while stop != 'stop':
    decision = input("What do you want to do(train, test, save, save test)? ")
    
    if decision == 'train':
        epoch = int(input('epoch: '))
        train_zip(raw_data, epoch)

    if decision == 'save':
        save_file = open('C:/Users/bense/Downloads/zipped_data.csv', 'w')
        epoch = int(input('epoch: '))
        save_string = ''
        
        for i in range(epoch):
            in_list  = array([[float(a) / 255] for a in raw_data[i].split(',')[1:785]])
            Enc_1.zip_data(in_list, 0) 
            
            zip_list = raw_data[i].split(',')[0]
            for a in range(0, num_a): zip_list += ',' + str(Enc_1.a_hid[a][0])
                  
            save_string += zip_list + '\n'
            
        save_file.write(save_string)  
        save_file.close()
        
    if decision == 'save test':
        save_file = open('C:/Users/bense/Downloads/zipped_data_test.csv', 'w')
        raw_file = open('C:/Users/bense/Downloads/mnist_test.csv', 'r')
        raw_test = raw_file.readlines()
        
        epoch = int(input('epoch: '))
        save_string = ''
        
        for i in range(epoch):
            in_list  = array([[float(a) / 255] for a in raw_test[i].split(',')[1:785]])
            Enc_1.zip_data(in_list, 0) 
            
            zip_list = raw_test[i].split(',')[0]
            for a in range(0, num_a): zip_list += ',' + str(Enc_1.a_hid[a][0])
                  
            save_string += zip_list + '\n'
            
        save_file.write(save_string)  
        save_file.close()
              
    if decision == 'test':
        raw_file = open('C:/Users/bense/Downloads/mnist_test.csv', 'r')
        raw_test = raw_file.readlines()
        
        sample_num = int(input("sample number: "))
        in_list = array([[float(a) / 255] for a in raw_test[sample_num].split(',')[1:785]])
        Enc_1.zip_data(in_list, 0, True, False)
        
        out_string = ''
        
        draw_image(in_list)
        print('\n')
        draw_image(Enc_1.a_out)


