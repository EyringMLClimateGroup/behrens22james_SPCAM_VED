"""
This script computes MSE and biases between a linear version of reference ANN (Rasp et al.,2018)
and test, validation, training data set 
"""

from tensorflow.keras.layers import Input, Dense
from cbrain.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import mse, binary_crossentropy

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, Callback


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


import tensorflow as tf
from cbrain.imports import *

from cbrain.utils import *
import pandas as ps
from cbrain.data_generator import DataGenerator

reference_input_shape=(94,)#based on reference ANN 
hidden_layer_shape=256
reference_output_shape=65 #based on reference ANN

inputs =Input(shape=reference_input_shape, name='reference_input')
x_0 =Dense(hidden_layer_shape, activation='linear')(inputs)
x_1 =Dense(hidden_layer_shape, activation='linear')(x_0)
x_2 =Dense(hidden_layer_shape, activation='linear')(x_1)
x_3 =Dense(hidden_layer_shape, activation='linear')(x_2)
x_4 =Dense(hidden_layer_shape, activation='linear')(x_3)
x_5 =Dense(hidden_layer_shape, activation='linear')(x_4)
x_6 =Dense(hidden_layer_shape, activation='linear')(x_5)
x_7 =Dense(hidden_layer_shape, activation='linear')(x_6)
x_8 =Dense(hidden_layer_shape, activation='linear')(x_7)
outputs=Dense(reference_output_shape, activation='linear')(x_8)

reference_lin = Model(inputs, outputs, name='reference_linear_model')
reference_lin.summary()


#load weights and biases of linear reference version of Rasp et al
reference_lin.load_weights('saved_models/reference_ANN_linear/reference_ANN_lin_ref.h5')




import pickle
# load Rasp output normalization
scale_dict_pnas= pickle.load(open('nn_config/scale_dicts/002_pnas_scaling.pkl','rb'))




in_vars = ['QBP', 'TBP','VBP','PS', 'SOLIN', 'SHFLX', 'LHFLX'] # like reference ANN use of merdional wind profiles 
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']

# test data with Rasp scaling 

val_gen_I = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_OND.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/001_norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_pnas,
    batch_size=8192,
    shuffle=True
)

## test data 
print('start Test evaluation')

time_max=4415
X_1=np.nan*np.zeros((time_max,8192,94))
Y_1=np.nan*np.zeros((time_max,8192,65))
Y_1_emul=np.nan*np.zeros((time_max,8192,65))

for i in tqdm(np.arange(X_1[:,1,1].size)):
    X_1[i],Y_1[i]=val_gen_I[i]
    Y_1_emul[i]=reference_lin.predict(X_1[i]) # predict with Rasp scaling
    
Y_1_1=np.reshape(Y_1,(time_max*8192,65))
Y_emul=np.reshape(Y_1_emul,(time_max*8192,65))

# normalization of absolute values with VAE scaling 
mse=np.mean((Y_emul[:,0:65]-Y_1_1[:,0:65])**2)
print(mse)

mean_bias=np.mean(Y_emul[:,0:65]-Y_1_1[:,0:65])
print(mean_bias)

del Y_1, Y_emul, X_1

## validation data 

print('start Validation evaluation')

## validation data with Rasp scaling
val_gen_III = DataGenerator(
    data_fn = '../preprocessed_data/1918_valid_3_month_AMJ.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/001_norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_pnas,
    batch_size=8192,
    shuffle=True
)


time_max=4367
val_X=np.nan*np.zeros((time_max,8192,94))
val_Y=np.nan*np.zeros((time_max,8192,65))
val_Y_emul=np.nan*np.zeros((time_max,8192,65))


for i in tqdm(np.arange(val_X[:,1,1].size)):
    val_X[i],val_Y[i]=val_gen_III[i]
    val_Y_emul[i]=reference_lin.predict(val_X[i])
   

val_Y_=np.reshape(val_Y,(time_max*8192,65))
val_Y_emul_=np.reshape(val_Y_emul,(time_max*8192,65))

val_mse=np.mean((val_Y_emul_[:,0:65]-val_Y_[:,0:65])**2)
print(val_mse)

val_mean_bias=np.mean(val_Y_emul_[:,0:65]-val_Y_[:,0:65])
print(val_mean_bias)
del val_Y, val_Y_emul, val_X

### test data 
print('start Training evaluation')

## test data with Rasp scaling
train_gen_III = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_JAS_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/001_norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_pnas,
    batch_size=8192,
    shuffle=True
)


time_max=4365


train_X=np.nan*np.zeros((time_max,8192,94))
train_Y=np.nan*np.zeros((time_max,8192,65))
train_Y_emul=np.nan*np.zeros((time_max,8192,65))


for i in tqdm(np.arange(train_X[:,1,1].size)):
    train_X[i],train_Y[i]=train_gen_III[i]
    train_Y_emul[i]=reference_lin.predict(train_X[i])
    

train_Y_=np.reshape(train_Y,(time_max*8192,65))
train_Y_emul_=np.reshape(train_Y_emul,(time_max*8192,65))

train_mse=np.mean((train_Y_emul_[:,0:65]-train_Y_[:,0:65])**2)
print(train_mse)

train_mean_bias=np.mean(train_Y_emul_[:,0:65]-train_Y_[:,0:65])
print(train_mean_bias)

## saving MSE and biases of test, validation and training data sets
Perf_array=ps.DataFrame({'test_mse':[mse],
                            'test_bias':[mean_bias],
                            'val_mse':[val_mse],
                            'val_bias':[val_mean_bias],
                            'train_mse':[train_mse],
                            'train_bias':[train_mean_bias]})

Perf_array.to_csv('perf_analy/reference_ANN_linear_model_PNAS_scaling.csv')
