# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:40:05 2023

@author: aaron
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

with open("data_intermediateinteracing.pkl",'rb') as f:
    Density_Data = pkl.load(f)

lower_i = np.tril_indices(6,k=-1)
upper_i = np.triu_indices(6,k=1)
dim_data = np.ma.size(Density_Data, axis=0)
dim_time = np.ma.size(Density_Data, axis=1)

def get_flat_elems(m):
    d = m.diagonal().real
    of = m[upper_i]
    re, im = of.real, of.imag
    return np.concatenate([d, re, im])
    
n = dim_data * dim_time
p = 36    
Flattened_Data = []
for i in np.arange(dim_data):
    for r in Density_Data[i]:
        flattened_r = get_flat_elems(r)
        Flattened_Data.append(flattened_r)

X = np.array(Flattened_Data)

U = np.zeros((p,1))
for j in np.arange(p):
    U[j] = np.mean(X[j])

h = np.ones((n,1))
B = X - h@(U.T) 
C = (1/(dim_data-1))*(B.T)@B # Covariance Matrix
Eig, V = np.linalg.eig(C)
plt.plot(np.arange(p),Eig,'ro')
plt.show()

g = np.sum(Eig)
L = 0; g_L = 0
ratio = 0
while (ratio < 0.95):
    g_L += Eig[L]
    ratio = g_L/g
    L += 1
print ("Cut off data point corresponding to L = ",L)

W = V[:,0:L]
T = np.dot(B,W)

plt.plot(np.arange(n),T[:,0],'bo')
plt.plot(np.arange(n),T[:,1],'ro')
plt.plot(np.arange(n),T[:,17],'go')
plt.plot(np.arange(n), np.dot(B,V)[:,35],'yo')
plt.show()