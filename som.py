# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:28:20 2020

@author: atidem
"""
## spyder da kodlanmıştır.
## epoch değerini değiştirebilirsiniz 

from math import exp,log
from random import random,shuffle,sample
import copy
import sys

#%% data prepare for train

data = []
transpose = []

with open("emlak-veri.txt") as fp:
    lines = fp.readlines()
    
# get data patrice
for line in lines:
    data.append(line.split(","))

# transpose data matrice
for i in range(len(data[1])):
    l = []
    for j in range(1,len(data)):
        try:
            l.append(float(data[j][i]))
        except ValueError:
            l.append(data[j][i])
    transpose.append(l)
    
# min max normalize
for i in range(len(data[1])) :
    if(i!=4):
        colmin = min(transpose[i])
        colmax = max(transpose[i])
        for j in range(1,len(data)):
            data[j][i] = (float(data[j][i]) - colmin) / (colmax - colmin)
        
# for categorical feature 
uniqCategory = []
for i in transpose[4]:
    if i not in uniqCategory:
        uniqCategory.append(i)

# dummy coding
encodingData= []
encodingData.append(uniqCategory)
for i in range(1,len(data)):
    row = []
    for j in range(len(uniqCategory)):
        if(uniqCategory[j] == data[i][4]):
            row.append(1)
        else:
            row.append(0)
    encodingData.append(row)

# modified ended for data
readyData = []
for i in range(len(data)):
    row = []
    for j in range(len(data[1])):
        if(j==4):
            pass
        else:
            row.append(data[i][j])
    for k in range(len(uniqCategory)):
        row.append(encodingData[i][k])
    readyData.append(row)

#%%
## n number of itr
## sigma parameter of neighbr
## qi initial sigma
## hij topological func
## dist distance 

"""sigma 0"""
qi = 1.0
t = 5000
timeConstant = 1000
itr = 5000
lrS = 0.1
lr = 0.1
som = []

"""10*10 , 16 boyutlu vector"""
def initSom():
    tmp = 1/100
    for i in range(100):
        vector = []
        vector.clear()
        for k in range(16):
            vector.append(random())
        tmp = tmp + (1/101)
        som.append(vector)

"""difference """    
def diff(x=[],y=[]):
    return list(map(lambda x,y: x-y ,x,y))

"""distance calc"""        
def dist(x=[],y=[]):
    dif = list(map(lambda x: x*x ,diff(x,y)))
    return sum(dif)**(1/2)

""" kapsama alanı"""
def sigma(n):
    return qi*exp(-n/t)
    
"""i winner neuron , j something else"""
def hij(n,i=[],j=[]):
    try:
        neigh = exp(-(dist(i,j)**2)/(2*(sigma(n)**2)))
        if(neigh<0):
            return 0
        else:
            return neigh
        return 
    except ZeroDivisionError:
        return 0

"""winner id return """
def winner(x=[]):
    msr = sys.maxsize
    winnerId = sys.maxsize
    for b in range(len(som)):
        d = dist(som[b],x)
        if(d<msr):
            winnerId = b
            msr = d
    if(winnerId==sys.maxsize):
        print("kimsesi yok bu kayıdın !!")
    return winnerId

"""delta weight calc , what a good calc aha"""
def updatedW(n,lr,winN=[],w=[],x=[]):        
    dW = list(map(lambda x: x * lr * hij(n,winN,w) , diff(x,w)))
    new = list(map(lambda x,y: x+y  ,w,dW))
    return new

#%% algorithm

initSom()
readyData.pop(0)

for a in range(itr):
    if(a%100==0):
        print(" {}.iterasyon Learning rate:{}".format(a,lr))
    inp = sample(readyData,1)[0]
    winVec = som[winner(inp)]
    for c in range(len(som)):
        if (float(qi*exp(-a/t)) > float(dist(winVec,som[c]))):
            #print(" kapsam {}  dist {}".format(qi*exp(-a/t),dist(winVec,som[c])))
            som[c] = copy.deepcopy(updatedW(a,lr,winVec,som[c],inp))
            
    if(lr >= 0.01):
        lr = lrS * exp(-a/timeConstant)
    

#%% ordering
## pdf deki gösterimde tüm noktalardan geçen tek çizgi görseli elde etmek için
                 
o = copy.deepcopy(som)  
orderList = []
first=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(o)):
    distList= []
    for j in range(len(o)):
        distList.append(dist(first,o[j]))
    ind = distList.index(min(distList))
    orderList.append(copy.deepcopy(o[ind]))
    first = copy.deepcopy(o[ind])
    o.pop(ind)
    
    
#%% visualize
import matplotlib.pyplot as plt
##plt.scatter(list(map(lambda x: x[0] ,som)),list(map(lambda x: x[2] ,som)))
#plt.hist2d(list(map(lambda x: x[0] ,som)),list(map(lambda x: x[1] ,som)))

"""x[1] şeklindeki ifadede bulunan sayılar değiştirilerek farklı değişkenler gözlemlenebilir."""
fig, ((axs0,axs1,axs2),(axs3,axs4,axs5),(axs6,axs7,axs8)) = plt.subplots(3, 3, figsize=(8, 8))
axs0.scatter(list(map(lambda x: x[5] ,som)),list(map(lambda x: x[4] ,som)))
axs1.scatter(list(map(lambda x: x[5] ,som)),list(map(lambda x: x[3] ,som)))
axs2.scatter(list(map(lambda x: x[5] ,som)),list(map(lambda x: x[6] ,som)))

axs3.plot(list(map(lambda x: x[5] ,orderList)),list(map(lambda x: x[2] ,orderList)),'o-')
axs4.plot(list(map(lambda x: x[5] ,orderList)),list(map(lambda x: x[3] ,orderList)),'o-')
axs5.plot(list(map(lambda x: x[5] ,orderList)),list(map(lambda x: x[4] ,orderList)),'o-')


axs6.hist2d(list(map(lambda x: x[5] ,som)),list(map(lambda x: x[6] ,som)))
axs7.hist2d(list(map(lambda x: x[5] ,som)),list(map(lambda x: x[7] ,som)))
axs8.hist2d(list(map(lambda x: x[5] ,som)),list(map(lambda x: x[8] ,som)))
plt.show()



