"""
this script computes the reproduction skills of SP varaibles of LR_clim_clim_conv
for test, validation, training data 
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



    
LR_clim_clim_conv.load_weights('./saved_models/LR_clim_clim_conv/LR_clim_clim_conv_40_opt_net.h5')

## test data = October, November, December of first year of SPCAM simulations 
val_gen_II = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_OND.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=8192,
    shuffle=True
)






time_max=4415
X=np.nan*np.zeros((time_max,8192,64))
Y=np.nan*np.zeros((time_max,8192,129))
Y_emul=np.nan*np.zeros((time_max,8192,129))

## test data 

print('start test pred')

for i in np.arange(X[:,1,1].size):
    X[i],Y[i]=val_gen_II[i]
    Y_emul[i]=LR_clim_clim_conv.predict(X[i])

    print(i)
    
X_=np.reshape(X,(time_max*8192,64))
Y_=np.reshape(Y,(time_max*8192,129))
Y_emul_=np.reshape(Y_emul,(time_max*8192,129))

print('End test pred')

mse=np.mean((Y_emul_[:,0:65]-Y_[:,0:65])**2)

print(mse)

mean_bias=np.mean((Y_emul_[:,0:65]-Y_[:,0:65]))
print(mean_bias)
del  X,Y, Y_emul
del X_,Y_,Y_emul_ 

# validation data 

print('start Validation evaluation')

val_gen_III = DataGenerator(
    data_fn = '../preprocessed_data/1918_valid_3_month_AMJ.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=8192,
    shuffle=True
)
time_max=4367
val_X=np.nan*np.zeros((time_max,8192,64))
val_Y=np.nan*np.zeros((time_max,8192,129))
val_Y_emul=np.nan*np.zeros((time_max,8192,129))


for i in np.arange(val_X[:,1,1].size):
    val_X[i],val_Y[i]=val_gen_III[i]
    val_Y_emul[i]=LR_clim_clim_conv.predict(val_X[i])
    print(i)
    
val_X_=np.reshape(val_X,(time_max*8192,64))
val_Y_=np.reshape(val_Y,(time_max*8192,129))
val_Y_emul_=np.reshape(val_Y_emul,(time_max*8192,129))

val_mse=np.mean((val_Y_emul_[:,0:65]-val_Y_[:,0:65])**2)
print(val_mse)
mean_bias_val=np.mean((val_Y_emul_[:,0:65]-val_Y_[:,0:65]))
print(mean_bias_val)
del val_X, val_Y, val_Y_emul
del val_X_,val_Y_,val_Y_emul_

## training data 

print('start Training evaluation')

train_gen_IV = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_JAS_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=8192,
    shuffle=True
)

time_max=4365

train_X=np.nan*np.zeros((time_max,8192,64))
train_Y=np.nan*np.zeros((time_max,8192,129))
train_Y_emul=np.nan*np.zeros((time_max,8192,129))
for i in np.arange(train_X[:,1,1].size):
    
    train_X[i],train_Y[i]=train_gen_IV[i]
    train_Y_emul[i]=LR_clim_clim_conv.predict(train_X[i])
    print(i)

train_X_=np.reshape(train_X,(time_max*8192,64))
train_Y_=np.reshape(train_Y,(time_max*8192,129))
train_Y_emul_=np.reshape(train_Y_emul,(time_max*8192,129))

train_mse=np.mean((train_Y_emul_[:,0:65]-train_Y_[:,0:65])**2)
print(train_mse)

mean_bias_train=np.mean((train_Y_emul_[:,0:65]-train_Y_[:,0:65]))
print(mean_bias_train)
del train_X, train_Y, train_Y_emul

# save MSE and biases of test, validation and training data in .csv 
Perf_array=ps.DataFrame({'test_mse':[mse],
                            'test_bias':[mean_bias],
                            'val_mse':[val_mse],
                            'val_bias':[mean_bias_val],
                            'train_mse':[train_mse],
                            'train_bias':[mean_bias_train]})

Perf_array.to_csv('perf_analy/LR_clim_clim_conv.csv')
    
    
