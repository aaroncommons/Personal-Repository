# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 12:45:23 2021

@author: aaron
This pyhton script plots the poisson distribution for various mean values and
performs a computer experiment by essentially throwing N darts randomly at L
uniform regions and analysing the probability that a given region has a 
particular number of darts
"""

import numpy as np
import matplotlib.pylab as plt
import numpy.random as ran
import math

print ("Name: Aaron Commons")
print ("Student ID: 20327617")

avg_n1 = 1
avg_n2 = 5
avg_n3 = 10

# Poisson Distribution Function
def poisson(n, avg):
    f = (avg**n / math.factorial(n)) * np.exp(-avg)
    return f

#N = np.arange(25)   #(Graphs not working for this initialisation)
k = 26
N = []; i = 0
while i < k:
    N.append(i)
    i += 1

# Plotting poisson distribution for different mean values
P1 = np.zeros(k); P2 = np.zeros(k); P3 = np.zeros(k)
for i in N:
    P1[i] = poisson(i, avg_n1)
    P2[i] = poisson(i, avg_n2)
    P3[i] = poisson(i, avg_n3)
    
plt.plot(N,P1,'r--o', markersize = 5, label = '$<n>=1$')
plt.plot(N,P2,'b--o', markersize = 5, label = '$<n>=5$')
plt.plot(N,P3,'g--o', markersize = 5, label = '$<n>=10$')
plt.xlabel("$n$")
plt.ylabel("$P(n)$")
plt.title("Poisson Distribution for Different Mean Numbers")
plt.legend(loc = 1)
plt.show()


# Sum of probablitities P(n) from n=0 to n=50
k1 = 51
N1 = []; i = 0
while i < k1:
    N1.append(i)
    i += 1

# Calling S1 = Σ P(n),  S2 = Σ nP(n) = <n>,  S3 = Σ n^2 P(n) = <n^2> 
S1_1 = 0; S1_2 = 0; S1_3 = 0
S2_1 = 0; S2_2 = 0; S2_3 = 0
S3_1 = 0; S3_2 = 0; S3_3 = 0

for n in N1:
    S1_1 = S1_1 + poisson(n, avg_n1)
    S1_2 = S1_2 + poisson(n, avg_n2)
    S1_3 = S1_3 + poisson(n, avg_n3)
    
    S2_1 = S2_1 + n*poisson(n, avg_n1)
    S2_2 = S2_2 + n*poisson(n, avg_n2)
    S2_3 = S2_3 + n*poisson(n, avg_n3)

    S3_1 = S3_1 + n**2*poisson(n, avg_n1)
    S3_2 = S3_2 + n**2*poisson(n, avg_n2)
    S3_3 = S3_3 + n**2*poisson(n, avg_n3)

# variance = <n^2> - <n>^2 = S3 - S2^2
var_1 = S3_1 - S2_1**2
var_2 = S3_2 - S2_2**2
var_3 = S3_3 - S2_3**2

# standard deviation = sqrt(var)
sd_1 = np.sqrt(var_1)
sd_2 = np.sqrt(var_2)
sd_3 = np.sqrt(var_3)

print ("Standard Deviation for mean 1 is", sd_1)
print ("Standard Deviation for mean 5 is", sd_2)
print ("Standard Deviation for mean 10 is", sd_3)

# printing data in table
d = [ ["1", S1_1, S2_1, S3_1, var_1, sd_1],
     ["5", S1_2, S2_2, S3_2, var_2, sd_2],
     ["10", S1_3, S2_3, S3_3, var_3, sd_3]]
 
print ("\n")    
print ("{:<8} {:<20} {:<20} {:<20} {:<20} {:<20}".format('Mean','Normalisation','<n>', '<n^2>', 
                                                         'Variance', 'Standard Deviation'))
for v in d:
    mean, S1, S2, S3, var, sd = v
    print ("{:<8} {:<20} {:<20} {:<20} {:<20} {:<20}".format( mean, S1, S2, S3, var, sd))
    
# Dart Experiment ######################################################################    

Ndart = 50 # number of darts thrown per trial
L = 100 # number of regions
ntrial = 10 # number of trials 
nmax = 26  # maximum number of darts per region to look for

# Finding number of elements with n occurences in array and adding to array H
#   Equivalent to finding number of regions with n darts

H = np.zeros(nmax)
for k in np.arange(ntrial): # Performing experiment ntrial times
    DartArray = ran.randint(1,L, size = Ndart) # Throwing N darts
    for n in np.arange(nmax): 
        h = 0  # stores number of times a particular integer appears n times in DartArray
        for i in np.arange(1, L+1):
            count = np.count_nonzero(DartArray == i) # Counts number of elements in array with value i
            if count == n:
                h = h + 1
        H[n] = H[n] + h

# Histogram of H(n)
plt.hist(H, bins = 7)
plt.show()

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
plt.xlabel("$n$", fontsize = 12)
plt.ylabel("$P_{sim}(n)$", fontsize = 12)
plt.title("Plot of Probability Distribution of Simulation")
plt.legend(loc = 1)
plt.show()

plt.plot(N,np.log(P),'r--o', markersize = 5, label = 'Poisson')
plt.plot(N,np.log(Psim),'bo', markersize = 5, label = 'Numerical Data')
plt.ylim((-17,0))
plt.xlabel("$n$", fontsize = 12)
plt.ylabel("$log[P_{sim}(n)]$", fontsize = 12)
plt.title("Plot of Probability Distribution of Simulation")
plt.legend(loc = 1)
plt.show()
