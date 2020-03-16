import random as rand
import numpy  as np 
from math import acos, pi

class CNN:
    def __init__(self):
        self.convWei               = np.array([[.5, .5, .5], [.5, .5, .5], [.5, .5, .5]])
        self.fullWei, self.fullBia = np.random.rand(10, 169) - .5, np.random.rand(10, 1) - .5      
    
        self.convCoor, self.convLayer = [], np.zeros((26, 26)) 
        self.poolLayer, self.outLayer = np.zeros((13, 13)), []
        
        for a in range(676):
            self.convCoor.append((a % 26, int(a / 26)))
        
        self.costFunc   = 0.0
        self.learnRate  = 2
        
    def activ(self, z, deriv=True):
        if deriv == True:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))
    
    def pool(self, forw):
        cL, a, r, c = self.convLayer, 0, 0, 0
        dConv, b    = np.zeros((26, 26)), 0 
        
        if forw == True:
            for a in range(169):
                if a != 0 and a % 13 == 0:
                    r, c = r + 2, 0
                elif a != 0:
                    c += 2      
                self.poolLayer[int(c/2)][int(r/2)] = 0
                
                for cA in range(c, c + 2):
                    for rA in range(r, r + 2):
                        if cL[cA, rA] > self.poolLayer[int(c/2)][int(r/2)]: self.poolLayer[int(c/2)][int(r/2)] = cL[cA, rA]
                    
        else:
            for a in range(169):
                if a != 0 and a % 13 == 0:
                    r, c = r + 2, 0
                elif a != 0:
                    c += 2 
                    
                for cA in range(c, c + 2):
                    for rA in range(r, r + 2):
                        dConv[cA, rA] = self.dPool[a]
            
            return dConv

    def feedFor(self, inValues, outCorrect, count):
        def convFor(inValues):
            x, y, i = 1, 1, inValues

            for a in range(1, 676):
                x, y = self.convCoor[a]
                
                boxList = np.array([[i[x-1][y-1][0], i[x][y-1][0], i[x+1][y-1][0]],
                                    [i[x-1][y][0]  , i[x][y][0]  , i[x+1][y][0]  ],
                                    [i[x-1][y+1][0], i[x][y+1][0], i[x+1][y+1][0]]])

                sumBox = sum(np.multiply(boxList, self.convWei).flatten())
                self.convLayer[a%26][int(a/26)] = self.activ(sumBox, False)
                       
        def fullFor():
            poolFlat      = self.poolLayer.flatten()
            poolFlat      = [[a] for a in poolFlat]
            self.outLayer = self.activ( np.dot(self.fullWei, poolFlat) + self.fullBia, False )

        convFor(inValues)
        self.pool(True)
        fullFor()
        
        if count % 10 == 0:
            for i in range(10): self.costFunc += (((self.outLayer[i] - outCorrect[i]) ** 2) / 10)     

    def backProp(self, outCorrect, inValues, count):     
        def convBackProp(inValues):
            daConv, tempConvWei = self.pool(False).flatten(), self.convWei
            x, y, i             = 1, 1, inValues 
            dzConv              = self.activ(self.convLayer.flatten())  
            
            for a in range(1, 676):
                x, y = self.convCoor[a]
                
                boxList = np.array([[i[x-1][y-1][0], i[x][y-1][0], i[x+1][y-1][0]],
                                    [i[x-1][y][0]  , i[x][y][0]  , i[x+1][y][0]  ],
                                    [i[x-1][y+1][0], i[x][y+1][0], i[x+1][y+1][0]]])

                tempConvWei -= np.multiply(((dzConv[a] * daConv[a]) * boxList), self.learnRate)
                
            self.convWei = tempConvWei
                           
        def fullBackProp(dA1, a1):
            dA0, dz1 = np.zeros(169), self.activ(a1)
            poolFlat = self.poolLayer.flatten()          
            
            for a in range(10):               
                dA0 = dA0 + dA1[a] * dz1[a] * self.fullWei[a] 
            
            self.fullWei -= np.multiply([(a * poolFlat) for a in (dA1 * dz1)], self.learnRate)
            self.fullBia -= np.multiply([a for a in (dA1 * dz1)]             , self.learnRate)
            
            return dA0
                
        self.dOut  = (self.outLayer - outCorrect)
        self.dPool = fullBackProp(self.dOut, self.outLayer)
        convBackProp(inValues)
        
        learnCount = count % 100
        self.learnRate = (2 - count/60000) - learnCount / 100
        
        
        #da0 = np.zeros(len(self.aList[num]))
        #        print(len(self.aList[num]), len(self.fullWei[num]))

        #        for a in range(len(self.aList[num+1])):
        #            da0 = da0 + da[a] * dz[a] * self.fullWei[num][a]