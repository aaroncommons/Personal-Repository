# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:40:03 2022
Python program which simulates the time evolution for a many body quantum mechanical system of four sites and two electrons, based on the Hubbard model
@author: aaron
"""

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm
import pickle

# 4 atom sites, 2 electrons
L = 4 # number of sites
N_e = 2 # number of electrons
N_H = int(math.factorial(L)/(math.factorial(N_e) * math.factorial(L-N_e))) # Hilbert space dim

# basis vectors - 6 ways to distribute 2 electrins over 4 sites
one = np.array([1,1,0,0])
two = np.array([1,0,1,0])
three = np.array([1,0,0,1])
four = np.array([0,1,1,0])
five = np.array([0,1,0,1])
six = np.array([0,0,1,1])

basis_vectors = np.array([one,two,three,four,five,six])
site_energies = np.array([10,0,5,15]) * 0.0
t = 1
V = 4

def Hamiltonian(B, E, t, V):
    """
    Parameters
    ----------
    B : Basis vectors 
    E : Site energies
    t : Transfer kinetic energy
    V : Interaction potential

    Returns
    -------
    Hamiltonian Matrix
    """
    N = np.size(B, axis=0) # Hilbert space dimension
    # Term 1 describes interaction of electrons with nuclei
    H_atom = np.identity(N) * (B @ E[:,None])
    
    # Term 2 corresponds to electron kinetic energy
    site_value = np.array([1,2,3,4])
    T = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(N):
            if np.abs(np.dot(site_value, B[i,:]) - np.dot(site_value, B[j,:])) == 1:
                T[i,j] = -t
            else:
                T[i,j] = 0
    
    # Term 3 describes interaction between electrons
    B_rolled = np.roll(B, -1, axis=1)
    B_rolled[:,-1] *= 0 # last col of rolled basis vectors are zero as first & last site in chain don't interact
    H_MB = V * np.identity(N) * (B @ B_rolled.T)
    
    return H_atom + T + H_MB

def ElectronDensity(psi, B):
    """
    psi : State vector
    B : Basis vectors
    -------
    Returns: Electron density at each site 
    """
    psi_sq = psi * psi.conj() 
    return psi_sq @ B

def DensityMatrix(psi):
    """
    psi : State vector
    -------
    Returns: Density matrix
    """
    return psi[:,None] @ psi.conj()[None,:]

def SiteOccupation(rho, B): 
    """
    rho : Density matrix
    B : Basis vectors
    -------
    Returns: Electron density at each site
    """
    rho_n = rho[:,:,None] * B
    return rho_n.trace()

H = Hamiltonian(basis_vectors, site_energies, t, V)
eigenergies, eigvectors = np.linalg.eigh(H)
E_gs = eigenergies[0]; psi_gs = eigvectors[:,0]
print ("Energy of ground state from smallest eigenvalue is", E_gs) 

density_gs = ElectronDensity(psi_gs, basis_vectors)
print ("Ground state site occupations are ", density_gs)

rho_gs = DensityMatrix(psi_gs)
# Density matrix checks
print ("Tr(rho) = ", rho_gs.trace())
print ("rho^2 - rho = \n", (rho_gs @ rho_gs) - rho_gs)
print ("rho^t - rho = \n", rho_gs.transpose()-rho_gs)
print ("Tr(H*rho) = E_gs = ", (H @ rho_gs).trace())
print ("Tr[n*rho] = ", SiteOccupation(rho_gs, basis_vectors))

##### Time dependance of density matrix #####

class LvN:
    def __init__(self, H_matrix, state_vector):
        """
        H_matrix : Hamiltonian of system
        state_vector : Initial state of system 
        """
        self.H = H_matrix
        self.psi = state_vector
        #initial density matrix for which we find time evolution
        self.rho = DensityMatrix(self.psi) 
        
    def LvNSol(self, time_interval):
        """
        Parameters
        ----------
        time_interval : Array of timesteps from t0 to tf
        Returns
        -------
        rhot : Time dependant density matrix from t0 to tf in system with Hamiltonian H
        """
        arg = 1j * (self.H[:, :, None]*time_interval)
        arg = arg.transpose((2,0,1))
        U = np.array([expm(-a) for a in arg])
        Ut = U.transpose((0,2,1)).conj()
        rhot = (U @ self.rho) @ Ut
        return rhot

# Time evolution of system observables for example initial state
t0 = 0; tf = 50; n = 500; # initial & final time & no. steps
tt = np.linspace(t0, tf, n)
#psi_ex = np.array([0.1,0,1,0,0,0])
psi_ex = np.random.uniform(size = (N_H))
psi_ex = psi_ex/np.linalg.norm(psi_ex)

S = LvN(H, psi_ex) # System object   
rhot = S.LvNSol(tt)

Trace = rhot.trace(axis1=1, axis2=2)
Occupation = np.array([SiteOccupation(r, basis_vectors) for r in rhot])
Total_Occupation = np.sum(Occupation, axis=1)

plt.plot(tt, Trace)
plt.title("Time Evolution of Trace of Density Matrix")
plt.xlabel("t")
plt.ylabel("Tr[rho(t)]")
plt.ylim(0,2)
plt.xlim(0,tf)
plt.show()

plt.plot(tt, Occupation[:,0], label='n1')  
plt.plot(tt, Occupation[:,1], label='n2')
plt.plot(tt, Occupation[:,2], label='n3')
plt.plot(tt, Occupation[:,3], label='n4')
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.xlim(0,tf)
plt.title("Time Evolution of Site Occupations")
plt.xlabel("t")
plt.ylabel("Average Occupancy")
plt.show()

plt.plot(tt, Total_Occupation)
plt.title("Time Evolution of Total Site Occupation")
plt.xlabel("t")
plt.ylabel("n(t)")
plt.ylim(1,3)
plt.xlim(0,tf)
plt.show()

#%% Generating data for machine learning

N = 5000 # no. of initial states for dataset
psi_uniform = np.random.uniform(size = (N_H,N))
psi_uniform = psi_uniform/np.linalg.norm(psi_uniform, axis=0)

n_T = 500 # no. timesteps
T0 = 0; TF = 50
TT = np.linspace(T0, TF, n_T)
   
ML_Data = np.array([LvN(H, psi_uniform[:,i]).LvNSol(TT) for i in np.arange(N)])
with open("data_intermediateinteracing_5000.pkl",'wb') as f:
    pickle.dump(ML_Data,f)
