# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:41:27 2023
Python program that builds neural networks to learn the time evolution of density matrix given a dataset of density matrix time series
@author: aaron
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%% import data
with open("data_intermediateinteracing_2000.pkl",'rb') as f:
    Density_Data = pkl.load(f)
    
#%% organising data
lower_i = np.tril_indices(6,k=-1)
upper_i = np.triu_indices(6,k=1)
n_samples_data = Density_Data.shape[0]
n_timesteps_data = Density_Data.shape[1]

f = 0.02
f_samples_data = 100 #int(f*n_samples_data)
Density_Data = Density_Data[:f_samples_data]

#%%
def get_flat_elems(m):
    d = m.diagonal().real
    off = m[upper_i]
    re, im = off.real, off.imag
    return np.concatenate([d, re, im])

def reconstruct_density_matrix(flat_m):
    d = flat_m[:6]
    off_real = flat_m[6:21]
    off_imag = flat_m[21:36]
    m = np.zeros((6,6), dtype='complex')
    m[upper_i] = off_real + 1j*off_imag
    m_hermitian = m.transpose().conj()
    m[lower_i] = m_hermitian[lower_i]
    m[np.diag_indices(6,2)] = d + 0*1j
    return m
    
n_elements = 36    
Flattened_Data = np.zeros((f_samples_data, n_timesteps_data, n_elements))
for i in np.arange(f_samples_data):
    for r in np.arange(n_timesteps_data):
        flattened_r = get_flat_elems(Density_Data[i,r])
        Flattened_Data[i,r] = flattened_r
        
#%% train/validate/test split
def split_dataset(dataset, n_steps_in, n_steps_out, n_steps_ahead):
    """
    Splits a given multivariate timeseries dataset into multiple input/output samples 
    where each sample has a specified number of input time steps and output timesteps.
    ----------
    dataset :  multivariate timeseries dataset of dimension (samples, timesteps, features)
    n_steps_in : number of input timesteps
    n_steps_out : subsequent number of output timesteps
    n_steps_ahead: number of timesteps ahead being predicted (=1 for next step)

    Returns
    -------
    X : input data array 
    y : output data array

    """
    n_samples_data = dataset.shape[0]
    n_timesteps_data = dataset.shape[1]
    X, y = list(), list()
    for j in range(n_samples_data):
        X_i, y_i = list(), list()
        for i in range(n_timesteps_data):
            end_ix = i + n_steps_in
            out_start_ix = end_ix + n_steps_ahead -1
            out_end_ix = out_start_ix + n_steps_out
            if out_end_ix > n_timesteps_data:
                break
            data_x, data_y = dataset[j,i:end_ix,:], dataset[j,out_start_ix:out_end_ix,:]
            X_i.append(data_x)
            y_i.append(data_y)
        X_i, y_i = np.array(X_i), np.array(y_i)
        X.append(X_i)
        y.append(y_i)
    X, y = np.array(X), np.array(y)
    return X, y

#%%
class model:
    def __init__(self, dataset, n_steps_ahead):
        """
        Parameters
        ----------
        dataset : Machine learning data
        n_steps_ahead : Number of steps ahead being predicted

        """
        self.dataset = dataset
        self.n_steps_ahead = n_steps_ahead 
        
    def organise_dataset(self, n_steps_in, n_steps_out):
        """
        Parameters
        ----------
        n_steps_in : Number of input steps
        n_steps_out : Number of output steps

        """
        X, y = split_dataset(self.dataset, n_steps_in, n_steps_out, self.n_steps_ahead)
        X = X.reshape(X.shape[0]*X.shape[1], n_steps_in, X.shape[3])
        y = y.reshape(y.shape[0]*y.shape[1], n_steps_out, y.shape[3])
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=(1/8), random_state=1)
        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]
    
    def build_model(self, n_steps_in, n_steps_out):
        Train_set, Validate_set, Test_set = self.organise_dataset(n_steps_in, n_steps_out)
        X_train, y_train = Train_set[0], Train_set[1]
        X_valid, y_valid = Validate_set[0], Validate_set[1]
        X_test, y_test = Test_set[0], Test_set[1]
        n_in, n_out = X_train.shape[2], y_train.shape[2]
        
        print ('Data shapes for number of steps ahead predicted being', self.n_steps_ahead)
        print ('Training: ', (X_train.shape, y_train.shape))
        print ('Validating: ', (X_valid.shape, y_valid.shape))
        print ('Testing: ', (X_test.shape, y_test.shape))
        
        y_train = y_train.reshape((y_train.shape[0], n_out))
        y_valid = y_valid.reshape((y_valid.shape[0], n_out))
        y_test = y_test.reshape((y_test.shape[0], n_out))
        
        nn_model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[n_steps_in, n_in]),
            #keras.layers.Dense(32, activation='relu'),
            #keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(n_out)
            ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        nn_model.compile(loss="mse", optimizer=optimizer)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        history = nn_model.fit(X_train, y_train, epochs=400,  batch_size=32,
                            validation_data=(X_valid, y_valid), callbacks=[early_stopping, mc])
        nn_model = keras.models.load_model('best_model.h5')
        nn_mse = nn_model.evaluate(X_test, y_test)
        return nn_model, nn_mse, history
        
    def plot_learning_curves(self, history):
        loss, val_loss = history.history["loss"], history.history["val_loss"]
        plt.plot(np.arange(len(loss)), loss, "b.-", label="Training loss")
        plt.plot(np.arange(len(val_loss)), val_loss, "r.-", label="Validation loss")
        #plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        #plt.axis([0, 50, 0, 0.6])
        plt.legend(fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        
        
#%% build model for various timesteps into future
n_steps_in = 1
n_steps_out = 1

n_steps_ahead = 1 
model_class = model(Flattened_Data, n_steps_ahead)
model, mse, history = model_class.build_model(n_steps_in, n_steps_out) 

#%%
model_class.plot_learning_curves(history)
plt.savefig("myImage.png", format="png", dpi=1200)
plt.show()
print ("Mean square error for the model is ", mse)
model.save('F1_new_model.h5')


