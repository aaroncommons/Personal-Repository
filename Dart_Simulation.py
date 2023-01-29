# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:54:39 2021

@author: aaron
"""
# Computer experiment using Python will be performed which is equivalent to throwing N darts at a board divided into L equally sized regions & 
# determining the probability that a region has a particular number of darts. 
# The aim is to compare the probability distribution of this experiment to the Poisson distribution and to observe the conditions which satisfy the requirements 
# for a Poisson distribution. 

import numpy as np
import matplotlib.pylab as plt
import numpy.random as ran
import math

# Poisson Distribution Function
def poisson(n, avg):
    f = (avg**n / math.factorial(n)) * np.exp(-avg)
    return f


Ndart = 50
L = 100
ntrial = 1000
nmax = 15   # maximum number of darts per region to look for

# Throwing N darts for ntrial experiments
DartArray = []
for i in np.arange(ntrial):
    DartArray.append(ran.randint(1,L, size = 50))

# Finding number of elements with n occurences in array and adding to array H
#   Equivalent to finding number of regions with n darts

H = np.zeros(nmax)
for k in np.arange(ntrial): # Performing experiment ntrial times
    DartArray = ran.randint(1,L, size = 50)
    for n in np.arange(nmax): 
        h = 0  # stores number of times a perticular integer appears n times in DartArray
        for i in np.arange(1, L+1):
            count = np.count_nonzero(DartArray == i) # Counts number of elements in array with value i
            if count == n:
                h = h + 1
        H[n] = H[n] + h

# Probability Distribution of H
sumH = L*ntrial
Psim = np.zeros(nmax)
for n in np.arange(nmax):
    Psim[n] = H[n]/sumH

# mean number of darts per region (50 darts, 100 regions -> 0.5 darts per region)
mean = Ndart/L

# Plotting Psim(n) for ntrial experiments
N = []; i = 0
while i < nmax:
    N.append(i)
    i += 1

P = np.zeros(nmax)   # poisson distribution with mean = 1/2
for i in N:
    P[i] = poisson(i, mean)
      
plt.plot(N,P,'r--o', markersize = 5, label = 'Poisson')
plt.plot(N,Psim,'bo', markersize = 5, label = 'Numerical Data')
plt.legend(loc = 1)
plt.show()
