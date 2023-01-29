# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 19:45:36 2021

@author: aaron
"""

import scipy.optimize as optimization
import numpy as np
import matplotlib.pylab as plt

print (" Name: Aaron Commons\n ID: 20327617")

days, cases = np.loadtxt("COVIDData.dat", skiprows=1, unpack=True)

# Plotting reported daily cases against day since pandemic onset
plt.plot(days, cases, 'bo', markersize=2.5)
plt.title("COVID-19 cases in the Republic of Ireland")
plt.xlabel("Days since February 29, 2020")
plt.ylabel("Number of reported daily cases")
plt.show()

# Plotting log of reported daily cases against day since pandemic onset
plt.plot(days, np.log(cases), 'bo', markersize=2.5)
plt.title("Natural logarithm of the number of cases vs time")
plt.xlabel("Days since February 29, 2020")
plt.ylabel("Natural logarithm of number of cases")
plt.show()

# Approximating the range of waves
firstWaveInc = np.arange(0,25)
firstWaveDec = np.arange(45,130)
secondWaveInc = np.arange(131,233)
secondWaveDec = np.arange(231,279)
thirdWaveInc = np.arange(280,314)
thirdWaveDec = np.arange(314,380)
fourthWaveInc = np.arange(475,530)

# fitting function
def func(x, a, b):     
    return a + b*x

# Determining fitting parameters for linear regression
def LinearRegression(WaveRange):
    fitparameter = optimization.curve_fit(func, days[WaveRange], np.log(cases[WaveRange]))[0]
    par1 = fitparameter[0]
    par2 = fitparameter[1]
    return par1, par2

# Determining linear fit parameters a, b for each line segment
a11, b11 = LinearRegression(firstWaveInc)
a12, b12 = LinearRegression(firstWaveDec)
a21, b21 = LinearRegression(secondWaveInc)
a22, b22 = LinearRegression(secondWaveDec)
a31, b31 = LinearRegression(thirdWaveInc)
a32, b32 = LinearRegression(thirdWaveDec)
a41, b41 = LinearRegression(fourthWaveInc)

# Plotting linear regressions on log(cases) vs time scale    
plt.plot(days, np.log(cases), 'co', markersize=2.5, label='data point')
plt.plot(days[firstWaveInc], func(days[firstWaveInc], a11, b11),'k--', linewidth=2)
plt.plot(days[firstWaveDec], func(days[firstWaveDec], a12, b12),'k--', linewidth=2)
plt.plot(days[secondWaveInc], func(days[secondWaveInc], a21, b21),'k--', linewidth=2)
plt.plot(days[secondWaveDec], func(days[secondWaveDec], a22, b22),'k--', linewidth=2)
plt.plot(days[thirdWaveInc], func(days[thirdWaveInc], a31, b31),'k--', linewidth=2)
plt.plot(days[thirdWaveDec], func(days[thirdWaveDec], a32, b32),'k--', linewidth=2)
plt.plot(days[fourthWaveInc], func(days[fourthWaveInc], a41, b41),'k--', linewidth=2, label='linear fit')
plt.title("Linear Regression of natural logarithm of number of cases vs time")
plt.xlabel("Days since February 29, 2020")
plt.ylabel("Natural logarithm of number of cases")
plt.legend(loc=2)
plt.show()

# function to find decay constant λ and initial case numbe n0
def expConstant (WaveRange, a, b):
    t0 = days[WaveRange[0]]
    λ = b
    n0 = np.exp(a + λ*t0)
    return λ, n0

# determining decay constant λ and initial case numbe n0 for each wave section
l11,n011 = expConstant(firstWaveInc, a11, b11)
l12,n012 = expConstant(firstWaveDec, a12, b12)
l21,n021 = expConstant(secondWaveInc, a21, b21)
l22,n022 = expConstant(secondWaveDec, a22, b22)
l31,n031 = expConstant(thirdWaveInc, a31, b31)
l32,n032 = expConstant(thirdWaveDec, a32, b32)
l41,n041 = expConstant(fourthWaveInc, a41, b41)

# printing decay constants
print ("\n")
print ("Decay cnstant for first wave increase is ", l11)
print ("Decay cnstant for first wave decrease is ", l12)
print ("Decay cnstant for second wave increase is ", l21)
print ("Decay cnstant for second wave decrease is ", l22)
print ("Decay cnstant for third wave increase is ", l31)
print ("Decay cnstant for third wave decrease is ", l32)
print ("Decay cnstant for fourth wave increase is ", l41)

# least square fit of exponential curves
expFit1 = n011*np.exp(l11*(days[firstWaveInc]-days[firstWaveInc[0]]))
expFit2 = n012*np.exp(l12*(days[firstWaveDec]-days[firstWaveDec[0]]))
expFit3 = n021*np.exp(l21*(days[secondWaveInc]-days[secondWaveInc[0]]))
expFit4 = n022*np.exp(l22*(days[secondWaveDec]-days[secondWaveDec[0]]))
expFit5 = n031*np.exp(l31*(days[thirdWaveInc]-days[thirdWaveInc[0]]))
expFit6 = n032*np.exp(l32*(days[thirdWaveDec]-days[thirdWaveDec[0]]))
expFit7 = n041*np.exp(l41*(days[fourthWaveInc]-days[fourthWaveInc[0]]))


# least square fit of exponential curves
plt.plot(days, cases, 'co', markersize=3, label='data point')
plt.plot(days[firstWaveInc],expFit1,'k--', linewidth=2)
plt.plot(days[firstWaveDec],expFit2,'k--', linewidth=2)
plt.plot(days[secondWaveInc],expFit3,'k--', linewidth=2)
plt.plot(days[secondWaveDec],expFit4,'k--', linewidth=2)
plt.plot(days[thirdWaveInc],expFit5,'k--', linewidth=2)
plt.plot(days[thirdWaveDec],expFit6,'k--', linewidth=2)
plt.plot(days[fourthWaveInc],expFit7,'k--', linewidth=2, label='exponential fit')
plt.title("Least square fits of number of cases vs time")
plt.xlabel("Days since February 29, 2020")
plt.ylabel("Number of reported daily cases")
plt.legend(loc=2)
plt.show()

# quadratic fit for fourth wave data
fourthWave = np.arange(490, 575)

def funct(x, a, b, c):     
    return a + b*x + c*x*x

quadparameter = optimization.curve_fit(funct, days[fourthWave], cases[fourthWave])[0]
quadpar1, quadpar2, quadpar3 = quadparameter

plt.plot(days[np.arange(400,575)], cases[np.arange(400,575)], 'co', markersize=5)
plt.plot(days[fourthWave], funct(days[fourthWave], quadpar1, quadpar2, quadpar3),'r--', linewidth=2, label='quadratic fit')
plt.title("Quadratic fit of fourth wave data")
plt.xlabel("Days since February 29, 2020")
plt.ylabel("Number of reported daily cases")
plt.legend(loc=2)
plt.show()

plt.plot(days[np.arange(400,575)], cases[np.arange(400,575)], 'co', markersize=5)
plt.plot(days[fourthWaveInc],expFit7,'k--', linewidth=2, label='exponential fit')
plt.title("Exponential fit of fourth wave data")
plt.xlabel("Days since February 29, 2020")
plt.ylabel("Number of reported daily cases")
plt.legend(loc=2)
plt.show()