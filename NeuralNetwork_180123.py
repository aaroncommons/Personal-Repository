# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:41:27 2023

@author: aaron
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

one = np.array([1,1,0,0])
two = np.array([1,0,1,0])
three = np.array([1,0,0,1])
four = np.array([0,1,1,0])
five = np.array([0,1,0,1])
six = np.array([0,0,1,1])
basis_vectors = np.array([one,two,three,four,five,six])

def SiteOccupation(rho, B): 
    rho_n = rho[:,:,None] * B
    return rho_n.trace()

#%% import data
with open("data_intermediateinteracing_5000.pkl",'rb') as f:
    Density_Data = pkl.load(f)
    
#%% organising data
lower_i = np.tril_indices(6,k=-1)
upper_i = np.triu_indices(6,k=1)
n_samples_data = Density_Data.shape[0]
n_timesteps_data = Density_Data.shape[1]

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
    m[lower_i] = off_real - 1j*off_imag
    m[np.diag_indices(6,2)] = d + 0*1j
    return m
    
n_elements = 36    
Flattened_Data = np.zeros((n_samples_data, n_timesteps_data, n_elements))
for i in np.arange(n_samples_data):
    for r in np.arange(n_timesteps_data):
        flattened_r = get_flat_elems(Density_Data[i,r])
        Flattened_Data[i,r] = flattened_r
        
#%% train/validate/test split
def split_dataset(dataset, n_steps_in, n_steps_out):
    """
    Splits a given multivariate timeseries dataset into multiple input/output samples 
    where each sample has a specified number of input time steps and output timesteps.
    ----------
    dataset :  multivariate timeseries dataset of dimension (samples, timesteps, features)
    n_steps_in : number of input timesteps
    n_steps_out : subsequent number of output timesteps

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
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > n_timesteps_data:
                break
            data_x, data_y = dataset[j,i:end_ix,:], dataset[j,end_ix:out_end_ix,:]
            X_i.append(data_x)
            y_i.append(data_y)
        X_i, y_i = np.array(X_i), np.array(y_i)
        X.append(X_i)
        y.append(y_i)
    X, y = np.array(X), np.array(y)
    return X, y

n_steps_in = 1
n_steps_out = 1
X, y = split_dataset(Flattened_Data, n_steps_in, n_steps_out)
n_timesteps = X.shape[1]
X = X.reshape(X.shape[0]*X.shape[1], n_steps_in, X.shape[3])
y = y.reshape(y.shape[0]*y.shape[1], n_steps_out, y.shape[3])

#%%
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=(1/8), random_state=1)

print ('Training: ', (X_train.shape, y_train.shape))
print ('Validating: ', (X_valid.shape, y_valid.shape))
print ('Testing: ', (X_test.shape, y_test.shape))
n_in, n_out = X.shape[2], y.shape[2]


#%% training LSTM model
lstm_model = keras.models.Sequential([
    keras.layers.LSTM(20, activation='relu', return_sequences=True, input_shape=[None, n_in]),
    keras.layers.LSTM(20, activation='relu', return_sequences=True),
    keras.layers.Dropout(rate=0.2),
    keras.layers.LSTM(20, activation='relu', return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(n_out))
])

optimizer = keras.optimizers.Adam(lr=0.0005)
lstm_model.compile(loss="mse", optimizer= optimizer)
lstm_model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = keras.callbacks.ModelCheckpoint('best_model_lstm.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history_lstm = lstm_model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_valid, y_valid), callbacks=[early_stopping, mc])

#%%
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)), loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)), val_loss, "r.-", label="Validation loss")
    #plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([0, 50, 0, 0.6])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
#%%
plot_learning_curves(history_lstm.history["loss"], history_lstm.history["val_loss"])
plt.show()

lstm_model = keras.models.load_model('best_model_lstm.h5')

lstm_mse = lstm_model.evaluate(X_test, y_test)
print ("LSTM mean_square_error: ", lstm_mse)

y_pred_lstm = lstm_model.predict(X_test)

#%% analysing model predictions
el = 50
time_series_act = Density_Data[el]
rho_0 = time_series_act[0]
Occupation_lstm_pred = [SiteOccupation(rho_0, basis_vectors)]
rho_0_flat = get_flat_elems(rho_0).reshape(1,1,n_elements)
for i in range(n_timesteps):
   rho_1_flat = lstm_model.predict(rho_0_flat) 
   rho_1 = reconstruct_density_matrix(rho_1_flat.reshape(n_elements))
   occ_1 = SiteOccupation(rho_1, basis_vectors)
   Occupation_lstm_pred.append(occ_1)
   rho_0_flat = rho_1_flat

#%%
Occupation_lstm_pred = np.array(Occupation_lstm_pred)
Occupation_act = np.array([SiteOccupation(r, basis_vectors) for r in time_series_act])
plt.plot(Occupation_act[:,2], label='actual')  
plt.plot(Occupation_lstm_pred[:,2], label='predicted')  
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.xlim(0,500)
plt.title("Time Evolution of Site Occupations")
plt.xlabel("t")
plt.ylabel("Average Occupancy")
plt.show()

#%% training a dense nn model
y_train = y_train.reshape((y_train.shape[0], n_elements))
y_valid = y_valid.reshape((y_valid.shape[0], n_elements))
y_test = y_test.reshape((y_test.shape[0], n_elements))

n_timesteps_in = X.shape[1]
dnn_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[n_steps_in, n_in]),
    #keras.layers.Dense(32, activation='relu'),
    #keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(n_out)
    ])

optimizer = keras.optimizers.Adam(lr=0.0001)
dnn_model.compile(loss="mse", optimizer=optimizer)
dnn_model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
mc = keras.callbacks.ModelCheckpoint('best_model_dnn.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history_dnn = dnn_model.fit(X_train, y_train, epochs=20,  batch_size=32,
                    validation_data=(X_valid, y_valid), callbacks=[early_stopping, mc])

plot_learning_curves(history_dnn.history["loss"], history_dnn.history["val_loss"])
plt.show()

dnn_model = keras.models.load_model('best_model_dnn.h5')

dnn_mse = dnn_model.evaluate(X_test, y_test)
print ("Dense NN mean_square_error: ", dnn_mse)

y_pred_dnn = dnn_model.predict(X_test)

#%% analysing dense neural network model predictions
el = 50
time_series_act = Density_Data[el]
rho_0 = time_series_act[0]
time_series_dnn = [rho_0]
rho_0_flat = get_flat_elems(rho_0).reshape(1,1,n_elements)
for i in range(n_timesteps):
   rho_1_flat = dnn_model.predict(rho_0_flat) 
   rho_1 = reconstruct_density_matrix(rho_1_flat.reshape(n_elements))
   time_series_dnn.append(rho_1)
   rho_0_flat = rho_1_flat
time_series_dnn = np.array(time_series_dnn)   

#%%
Occupation_act = np.array([SiteOccupation(r, basis_vectors) for r in time_series_act])
Occupation_dnn_pred = [SiteOccupation(r, basis_vectors) for r in time_series_dnn]

Occupation_dnn_pred = np.array(Occupation_dnn_pred)
plt.plot(Occupation_act[:,2], label='actual')  
plt.plot(Occupation_dnn_pred[:,2], label='predicted')  
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.xlim(0,500)
plt.title("Time Evolution of Site Occupations")
plt.xlabel("t")
plt.ylabel("Average Occupancy")
plt.show()

Trace_dnn_pred = time_series_dnn.trace(axis1=1, axis2=2)
plt.plot(Trace_dnn_pred, 'tab:orange', label='predicted')
plt.title("Time Evolution of Trace of Density Matrix")
plt.xlabel("t")
plt.ylabel("Tr[rho(t)]")
plt.legend(loc='upper right')
plt.ylim(0,2)
plt.show()

Total_Occupation_dnn_pred = np.sum(Occupation_dnn_pred, axis=1)
plt.plot(Total_Occupation_dnn_pred, 'tab:orange', label='predicted')
plt.title("Time Evolution of Total Site Occupation")
plt.xlabel("t")
plt.ylabel("n(t)")
plt.legend(loc='upper right')
plt.ylim(1,3)
plt.show()
