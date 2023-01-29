# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:10:56 2021

@author: aaron
"""
# Python program using three different numerical schemes; the simple Euler method, the improved Euler method, and the fourth order Runge-Kutta method,
# to analyse the numerical solution of an ODE.


import numpy as np
import matplotlib.pylab as plt

print ("Name: Aaron Commons")
print ("Student ID: 20327617")

def f(t, x):
    return (1+t)*x + 1 - 3*t + t**2

T, X = np.meshgrid(np.arange(0, 5, 0.2), np.arange(-3, 3, 0.24))

U = 1.0
V = f(T, X)
# Normalising the directional arrows
N = np.sqrt(U**2 + V**2)
U = U/N
V = V/N

fig, ax = plt.subplots()
q = ax.quiver(T, X, U, V)
plt.title("Directional Field of ODE on x-t Plane")
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()

# simple euler
def seuler(t,x,step):
	x_new = x + step*f(t,x)
	return x_new

# improved euler
def ieuler(t,x,step):
	x_new = x + 0.5*step*( f(t,x) + f(t + step, x + step*f(t, x)) )
	return x_new

# runge kutta
def rk(t,x,step):
	k1 = f(t,x)
	k2 = f(t + 0.5*step, x + 0.5*step*k1)
	k3 = f(t + 0.5*step, x + 0.5*step*k2)
	k4 = f(t + step, x + step*k3)
	x_new = x + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4) 
	return x_new

step = 0.04
start = 0.0
end = 5.0
#x0 = 0.065923
x0 = 0.0655
n = int((end - start)/step)

t = np.arange(start,end,step)
seul = np.zeros(n)
ieul = np.zeros(n)
ruku = np.zeros(n)

seul[0] = x0
ieul[0] = x0
ruku[0] = x0

for i in range(1,n):
	seul[i] = seuler(t[i-1], seul[i-1], step)
	ieul[i] = ieuler(t[i-1], ieul[i-1], step)
	ruku[i] = rk(t[i-1], ruku[i-1], step)
    
   
fig, ax = plt.subplots()
q = ax.quiver(T, X, U, V)
plt.plot(t, seul, "r", label = "Simple Euler")
plt.ylim((-3, 3))
plt.legend(loc = 3)
plt.title("Simple Euler Solution to ODE for $x(0)=0.0655$")
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()
  
   
fig, ax = plt.subplots()
q = ax.quiver(T, X, U, V)
plt.plot(t, seul, "r", label = "Simple Euler")
plt.plot(t, ieul, "b",  label = "Improved Euler")
plt.plot(t, ruku, "g", label = "Runge Kutta")
plt.ylim((-3, 3))
plt.legend(loc = 3)
plt.title("Numerical Solutions to ODE for $x(0)=0.0655$")
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()


