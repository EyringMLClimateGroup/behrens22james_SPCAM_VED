"""
This script is for the training of a linear version of the reference ANN of Rasp et al., 2018
"""
from tensorflow.keras.layers import Input, Dense
from cbrain.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse

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

## construct the linear reference model 
reference_input_shape=(94,) # same dimensions of input data as reference ANN
hidden_layer_shape=256
reference_output_shape=65 # same dim of output data as reference ANN 

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





import pickle
# load the native Rasp et al. 2018  output data scaling
scale_dict_pnas= pickle.load(open('nn_config/scale_dicts/002_pnas_scaling.pkl','rb'))

in_vars = ['QBP', 'TBP','VBP','PS', 'SOLIN', 'SHFLX', 'LHFLX'] 
#like reference ANN this network uses meridional wind profiles in contrast 
#to VAEs, LR_clim_clim_conv and AE_clim_clim_conv
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']

# training data consists of July, August, September of first year of SPCAM simulation
#(shuffled in space and time)
train_gen_III = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_JAS_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/001_norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_pnas,
    batch_size=1024,
    shuffle=True
)
# validation data consists of April, May and June of second year of SPCAM simulations 
val_gen_III = DataGenerator(
    data_fn = '../preprocessed_data/1918_valid_3_month_AMJ.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/001_norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_pnas,
    batch_size=1024,
    shuffle=True
)
# use learning rate and batch size based on Rasp et al., 2018
lr_start_rate=0.001


def schedule(epoch):
    
  if epoch < 7:
    return lr_start_rate
  elif epoch < 14:
    return lr_start_rate/5

  elif epoch < 21:
    return lr_start_rate/25

  elif epoch < 28:
     
    return lr_start_rate/125

  elif epoch < 35:
        
    return lr_start_rate/625

  else:
    return lr_start_rate/3125
    
 
reference_lin.compile(tf.keras.optimizers.Adam(lr=1E-4), loss=mse, metrics=['mse'])
callback_lr=LearningRateScheduler(schedule,verbose=1)
reference_lin.fit((train_gen_III),validation_data=(val_gen_III,None),epochs=40,shuffle=False,callbacks=
                              [callback_lr])


reference_lin.save_weights('saved_models/reference_ANN_linear/reference_ANN_lin_ref.h5')
reference_lin.save_weights('saved_models/reference_ANN_linear/reference_ANN_lin_ref.tf')


hist_df = ps.DataFrame(reference_lin.history.history) 

hist_csv_file = 'saved_models/reference_ANN_linear/reference_ANN_lin_ref_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

    
    
    
    
    

