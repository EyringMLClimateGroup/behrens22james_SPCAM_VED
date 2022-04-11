"""
training of LR_clim_clim_conv baseline
"""

from tensorflow.keras.layers import Input, Dense
from cbrain.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler,Callback


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


import tensorflow as tf
from cbrain.imports import *

from cbrain.utils import *
import pandas as ps

# constuction of LR_clim_clim_conv closely mirroring AE_clim_clim_conv with linear activations 

    
original_dim_input=64  # CAM variables node size 

original_dim_output=int(65+64) # SP + CAM variables node size 


# network parameters
input_shape = (original_dim_input,)
out_shape=(original_dim_output,)
intermediate_dim = 463 # number of first hidden layers of linear Encoder or last hidden layers of linear Decoder
batch_size = 714
latent_dim = 5 # latent space width
epochs = 40    
    
## Linear Encoder 
inputs =Input(shape=input_shape, name='encoder_input')
x_0 =Dense(intermediate_dim, activation='linear')(inputs)
x_1 =Dense(intermediate_dim, activation='linear')(x_0)
x_2 =Dense(int(np.round(intermediate_dim/2)), activation='linear')(x_1)
x_3 =Dense(int(np.round(intermediate_dim/4)), activation='linear')(x_2)
x_4 =Dense(int(np.round(intermediate_dim/8)), activation='linear')(x_3)
x_5 =Dense(int(np.round(intermediate_dim/16)), activation='linear')(x_4)



z_lin = Dense(latent_dim, activation='linear', name='z_lin')(x_5)





# instantiate encoder model
encoder_lin = Model(inputs, [z_lin], name='encoder_lin')


## linear Decoder
decoder_inputs =Input(shape=(latent_dim,), name='decoder_input')
x_1 =Dense(int(np.round(intermediate_dim/16)), activation='linear')(decoder_inputs)
x_2 =Dense(int(np.round(intermediate_dim/8)), activation='linear')(x_1)
x_3 =Dense(int(np.round(intermediate_dim/4)), activation='linear')(x_2)
x_4 =Dense(int(np.round(intermediate_dim/2)), activation='linear')(x_3)
x_5 =Dense(intermediate_dim, activation='linear')(x_4)
x_6 =Dense(intermediate_dim, activation='linear')(x_5)

outputs = Dense(original_dim_output, activation='linear')(x_6)

decoder_lin = Model(decoder_inputs, outputs, name='decoder')


emul_outputs=decoder_lin(encoder_lin(inputs))



LR_clim_clim_conv=Model(inputs,emul_outputs)

#loading scaling dictionary of SP variables 
scale_array=ps.read_csv('nn_config/scale_dicts/Scaling_cond_VAE.csv')


PHQ_std_surf=scale_array.PHQ_std.values[-1]

TPHYSTND_std_23=scale_array.TPHYSTND_std.values[-1]

PRECT_std=scale_array.PRECT_std.values
FSNS_std=scale_array.FSNS_std.values
FSNT_std=scale_array.FSNT_std.values
FLNS_std=scale_array.FLNS_std.values
FLNT_std=scale_array.FLNT_std.values

# loading scaling dictionaries of CAM variables 
scale_array_2D=ps.read_csv('nn_config/scale_dicts/Scaling_enc_II_range_profiles.csv')
scale_array_1D=ps.read_csv('nn_config/scale_dicts/Scaling_enc_II_range.csv')

TBP_std_surf=scale_array_2D.TBP_std.values[-1]

QBP_std_surf=scale_array_2D.QBP_std.values[-1]

Q_lat_std_surf=scale_array_1D.Q_lat_std.values

Q_sens_std_surf=scale_array_1D.Q_sens_std.values


Q_solar_std_surf=scale_array_1D.Q_sol_std.values

PS_std_surf=scale_array_1D.PS_std.values



# resulting output normalization dictionary 
scale_dict_II = {
    'PHQ': 1/PHQ_std_surf, 
    'QBP':1/QBP_std_surf,
    'TPHYSTND': 1/TPHYSTND_std_23, 
    'TBP':1/TBP_std_surf,
    'FSNT': 1/FSNT_std, 
    'FSNS': 1/FSNS_std, 
    'FLNT': 1/FLNT_std, 
    'FLNS': 1/FLNS_std, 
    'PRECT': 1/PRECT_std, 
    'LHFLX': 1/Q_lat_std_surf, 
    'SHFLX': 1/Q_sens_std_surf, 
    'SOLIN': 1/Q_solar_std_surf,
    'PS':1/PS_std_surf
}

in_vars = ['QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT','QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']

# Takes representative value for PS since purpose is normalization
PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];




from cbrain.data_generator import DataGenerator

# training data shuffled in space and time --> coming from July, August, September of first year of SPCAM simulations
train_gen = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_JAS_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=714,
    shuffle=True
)


# validation data AMJ (3 month) of second year of SPCAM simulation

val_gen = DataGenerator(
    data_fn = '../preprocessed_data/1918_valid_3_month_AMJ.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=714,
    shuffle=True
)
# learning rate schedule 
def schedule(epoch):
    
  if epoch < 7:
    return 0.00074594
  elif epoch < 14:
    return 0.00074594/5

  elif epoch < 21:
    return 0.00074594/25

  elif epoch < 28:
     
    return 0.00074594/125

  elif epoch < 35:
        
    return 0.00074594/625

  else:
    return 0.00074594/3125
    
 
LR_clim_clim_conv.compile(tf.keras.optimizers.Adam(lr=1E-4), loss=mse, metrics=['mse'])
callback_lr=LearningRateScheduler(schedule,verbose=1)
LR_clim_clim_conv.fit((train_gen),validation_data=(val_gen,None),epochs=40,shuffle=False,callbacks=
                              [callback_lr])


LR_clim_clim_conv.save_weights('saved_models/LR_clim_clim_conv/LR_clim_clim_conv_40_opt_net.h5')
LR_clim_clim_conv.save_weights('saved_models/LR_clim_clim_conv/LR_clim_clim_conv_40_opt_net.tf')

hist_df = ps.DataFrame(LR_clim_clim_conv.history.history) 

hist_csv_file = 'saved_models/LR_clim_clim_conv/LR_clim_clim_conv_40_opt_net.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

