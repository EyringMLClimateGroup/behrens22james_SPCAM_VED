"""
This notebook computes the R² of 700hPa predicted dT/dt and dq/dt of LR_clim_clim_conv
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

LR_clim_clim_conv.load_weights('./saved_models/LR_clim_clim_conv/LR_clim_clim_conv_40_opt_net.h5')

lat=np.arange(-90,90,180/64)
lon=np.arange(-180,180,360/128)
time=4415

X=np.nan*np.zeros((4415,8192,64))
Y=np.nan*np.zeros((4415,8192,129))
for i in np.arange(X[:,1,1].size):
    X[i],Y[i]=val_gen_II[i]

X_=np.reshape(X,(4415*8192,64))
Y_=np.reshape(Y,(4415*8192,129))



Y_emul=val_gen_II.output_transform.inverse_transform(LR_clim_clim_conv.predict(X_))
Y_real=val_gen_II.output_transform.inverse_transform(Y_)
lat=np.arange(-90,90,180/64)
lon=np.arange(-180,180,360/128)
time=4415

print('compute_r²')
T_tend_real=np.reshape(Y_real[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
T_tend_emul=np.reshape(Y_emul[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_real=np.reshape(Y_real[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_emul=np.reshape(Y_emul[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))

T_tend_real_long_mean=np.mean(T_tend_real,2)
T_tend_emul_long_mean=np.mean(T_tend_emul,2)
Q_tend_real_long_mean=np.mean(Q_tend_real,2)
Q_tend_emul_long_mean=np.mean(Q_tend_emul,2)

## dT/dt ID: 30:60
## dq/dt ID: 0:30


T_tend_real=np.reshape(Y_real[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
T_tend_emul=np.reshape(Y_emul[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_real=np.reshape(Y_real[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_emul=np.reshape(Y_emul[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))

T_tend_real_long_mean=np.mean(T_tend_real,2)
T_tend_emul_long_mean=np.mean(T_tend_emul,2)
Q_tend_real_long_mean=np.mean(Q_tend_real,2)
Q_tend_emul_long_mean=np.mean(Q_tend_emul,2)

lat=np.arange(-90,90,180/64)
lon=np.arange(-180,180,360/128)
time=4415

T_tend_R_2_700=np.nan*np.zeros((lat.size,lon.size))
Q_tend_R_2_700=np.nan*np.zeros((lat.size,lon.size))



# compute R2 values on level 20 ~ 700hPa

T_tend_R_2_700=1-np.mean((np.squeeze(T_tend_real[:,:,:,20])-np.squeeze(T_tend_emul[:,:,:,20]))**2,0)/np.var(np.squeeze(T_tend_real[:,:,:,20]),0)
Q_tend_R_2_700=1-np.mean((np.squeeze(Q_tend_real[:,:,:,20])-np.squeeze(Q_tend_emul[:,:,:,20]))**2,0)/np.var(np.squeeze(Q_tend_real[:,:,:,20]),0)

np.save('R_2_val/LR_clim_clim_conv_T_tend_R_2_700',T_tend_R_2_700)
np.save('R_2_val/LR_clim_clim_conv_Q_tend_R_2_700',Q_tend_R_2_700)




