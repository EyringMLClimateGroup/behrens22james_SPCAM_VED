"""

This script computes R² of 700hPa of dT/dt and dq/dt of VAE_clim_clim_conv.
Furthermore the R² values for the entire dT/dt and dq/dt profiles are computed 
and global mean R² profiles of dT/dt and dq/dt.
"""
from tensorflow.keras.layers import Lambda, Input, Dense
from cbrain.layers import *
from tensorflow.keras.models import Model
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

# reparameterization trick of VAE 
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    based on VAE presented on keras webpage for keras version 1 /
    recent keras VAE version can be seen on
    https://keras.io/examples/generative/vae/
    """

    z_mean, z_log_var = args
    batch= K.shape(z_mean)[0]
    dim=K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon=K.random_normal(shape=(batch,dim)) # epsilion= random_normal distributed tensor
    sample_prob=z_mean+K.exp(0.5*z_log_var)*epsilon #exp= elementwise exponential
    return sample_prob

# kl annealing to improve reproduction skills of VAE 
klstart = 2
# number of epochs over which KL scaling is increased from 0 to 1
kl_annealtime = 5

class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight
    def on_epoch_end (self, epoch, logs={}):
        if epoch > klstart :
            new_weight = min(K.get_value(self.weight) + (1./kl_annealtime), 1.)
            K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))


# the starting value of weight is 0
# define it as a keras backend variable
weight = K.variable(0.)

    
original_dim_input=64  # input node size (CAM variables)

original_dim_output=int(65+64) # output node size (SP + CAM variables)


# network parameters
input_shape = (original_dim_input,)
out_shape=(original_dim_output,)
intermediate_dim = 463 # nodes in first hidden layers of encoder and last hidden layers of decoder 
batch_size = 714
latent_dim = 5 # latent space dimensions
epochs = 40    
    
## Encoder 
inputs =Input(shape=input_shape, name='encoder_input')
x_0 =Dense(intermediate_dim, activation='relu')(inputs)
x_1 =Dense(intermediate_dim, activation='relu')(x_0)
x_2 =Dense(int(np.round(intermediate_dim/2)), activation='relu')(x_1)
x_3 =Dense(int(np.round(intermediate_dim/4)), activation='relu')(x_2)
x_4 =Dense(int(np.round(intermediate_dim/8)), activation='relu')(x_3)
x_5 =Dense(int(np.round(intermediate_dim/16)), activation='relu')(x_4)



z_mean = Dense(latent_dim, name='z_mean')(x_5)
z_log_var = Dense(latent_dim, name='z_log_var')(x_5)



# reparametrization trick
z = Lambda(sampling, output_shape=(latent_dim), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


## Decoder
decoder_inputs =Input(shape=(latent_dim,), name='decoder_input')
x_1 =Dense(int(np.round(intermediate_dim/16)), activation='relu')(decoder_inputs)
x_2 =Dense(int(np.round(intermediate_dim/8)), activation='relu')(x_1)
x_3 =Dense(int(np.round(intermediate_dim/4)), activation='relu')(x_2)
x_4 =Dense(int(np.round(intermediate_dim/2)), activation='relu')(x_3)
x_5 =Dense(intermediate_dim, activation='relu')(x_4)
x_6 =Dense(intermediate_dim, activation='relu')(x_5)

outputs = Dense(original_dim_output, activation='elu')(x_6)

decoder = Model(decoder_inputs, outputs, name='decoder')
decoder.summary()

emul_outputs=decoder(encoder(inputs)[2])

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
VAE_loss = K.mean(kl_loss*weight)


VAE_clim_clim_conv=Model(inputs,emul_outputs)
VAE_clim_clim_conv.add_loss(VAE_loss)
VAE_clim_clim_conv.add_metric(kl_loss, name='kl_loss', aggregation='mean')


#loading the output normalization scalars for SP variables ( stds over 3 months of SP simulation)

scale_array=ps.read_csv('nn_config/scale_dicts/Scaling_cond_VAE.csv')


PHQ_std_surf=scale_array.PHQ_std.values[-1]

TPHYSTND_std_23=scale_array.TPHYSTND_std.values[-1]# for dT/dt we are using the std on level 23 ~ 845 hPa

PRECT_std=scale_array.PRECT_std.values
FSNS_std=scale_array.FSNS_std.values
FSNT_std=scale_array.FSNT_std.values
FLNS_std=scale_array.FLNS_std.values
FLNT_std=scale_array.FLNT_std.values

# and the CAM variables 
scale_array_2D=ps.read_csv('nn_config/scale_dicts/Scaling_enc_II_range_profiles.csv')
scale_array_1D=ps.read_csv('nn_config/scale_dicts/Scaling_enc_II_range.csv')

TBP_std_surf=scale_array_2D.TBP_std.values[-1]

QBP_std_surf=scale_array_2D.QBP_std.values[-1]

Q_lat_std_surf=scale_array_1D.Q_lat_std.values

Q_sens_std_surf=scale_array_1D.Q_sens_std.values


Q_solar_std_surf=scale_array_1D.Q_sol_std.values

PS_std_surf=scale_array_1D.PS_std.values


# defining the scaling dict for the VAE training 

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

## CAM variables
#QBP = specific humidity
#TBP = temperature 
#PS = surface pressure 
#SOLIN = solar insolation
#SHFLX = surface sensible heat flux 
#LHFLX = surface latent heat flux

##  SP variables=
#PHQ = specific humidity tendency 
#TPHYSTND = temperature tendency 
#FSNT = shortwave heat flux model top
#FSNS = shortwave heat flux model surface 
#FLNT = longwave heat flux model top (OLR)
#FLNS = longwave heat flux model surface 
#PRECT = precipitation rate 

# Takes representative value for PS since purpose is normalization
PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];


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


VAE_clim_clim_conv.load_weights('./saved_models/VAE_clim_clim_conv/VAE_clim_clim_conv_BN5_40_opt_anneal.h5')


X=np.nan*np.zeros((4415,8192,64))
Y=np.nan*np.zeros((4415,8192,129))
for i in np.arange(X[:,1,1].size):
    X[i],Y[i]=val_gen_II[i]


X_=np.reshape(X,(4415*8192,64))
Y_=np.reshape(Y,(4415*8192,129))

print('Start VAE_pred')
Y_emul=val_gen_II.output_transform.inverse_transform(VAE_clim_clim_conv.predict(X_))
Y_real=val_gen_II.output_transform.inverse_transform(Y_)
lat=np.arange(-90,90,180/64)
lon=np.arange(-180,180,360/128)
time=4415

##dT/dt var ID [30:60] in val_gen_II
##dq/dt var ID [0:30] in val_gen_II

print('Start R² calc')
T_tend_real=np.reshape(Y_real[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
T_tend_emul=np.reshape(Y_emul[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_real=np.reshape(Y_real[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_emul=np.reshape(Y_emul[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))


## calculate R² on 700 hPa for dT/dt and dq/dt 
# 700 hPa ~ model lev 20
#print(P[20])

T_tend_R_2_700=np.nan*np.zeros((lat.size,lon.size))
Q_tend_R_2_700=np.nan*np.zeros((lat.size,lon.size))


T_tend_R_2_700=1-np.mean((np.squeeze(T_tend_real[:,:,:,20])-np.squeeze(T_tend_emul[:,:,:,20]))**2,0)/np.var(np.squeeze(T_tend_real[:,:,:,20]),0)
Q_tend_R_2_700=1-np.mean((np.squeeze(Q_tend_real[:,:,:,20])-np.squeeze(Q_tend_emul[:,:,:,20]))**2,0)/np.var(np.squeeze(Q_tend_real[:,:,:,20]),0)


np.save('R_2_val/VAE_clim_clim_conv_T_tend_R_2_700',T_tend_R_2_700)
np.save('R_2_val/VAE_clim_clim_conv_Q_tend_R_2_700',Q_tend_R_2_700)


##calculate R² for entire dT/dt and dq/dt profiles 
T_tend_R_2_glob=np.nan*np.zeros((lat.size,lon.size,30))
Q_tend_R_2_glob=np.nan*np.zeros((lat.size,lon.size,30))

for z in tqdm(np.arange(T_tend_R_2_glob[1,1,:].size)):
    T_tend_R_2_glob[:,:,z]=1-np.mean((np.squeeze(T_tend_real[:,:,:,z])-np.squeeze(T_tend_emul[:,:,:,z]))**2,0)/np.var(np.squeeze(T_tend_real[:,:,:,z]),0)
    Q_tend_R_2_glob[:,:,z]=1-np.mean((np.squeeze(Q_tend_real[:,:,:,z])-np.squeeze(Q_tend_emul[:,:,:,z]))**2,0)/np.var(np.squeeze(Q_tend_real[:,:,:,z]),0)
    

## calculate global mean R² profiles based on mean of sum of squared errors and variances of dT/dt and dq/dt 
T_tend_R_2_glob_mean=np.nan*np.zeros((30))
Q_tend_R_2_glob_mean=np.nan*np.zeros((30))

for z in tqdm(np.arange(T_tend_R_2_glob_mean.size)):
    T_tend_R_2_glob_mean[z]=1-np.mean(np.mean(np.mean((np.squeeze(T_tend_real[:,:,:,z])-np.squeeze(T_tend_emul[:,:,:,z]))**2,2),1),0)/np.mean(np.mean(np.var(np.squeeze(T_tend_real[:,:,:,z]),0),1),0)
    Q_tend_R_2_glob_mean[z]=1-np.mean(np.mean(np.mean((np.squeeze(Q_tend_real[:,:,:,z])-np.squeeze(Q_tend_emul[:,:,:,z]))**2,2),1),0)/np.mean(np.mean(np.var(np.squeeze(Q_tend_real[:,:,:,z]),0),1),0)
    

np.save('R_2_val/VAE_clim_clim_conv_T_tend_R_2_glob',T_tend_R_2_glob)
np.save('R_2_val/VAE_clim_clim_conv_Q_tend_R_2_glob',Q_tend_R_2_glob)
np.save('R_2_val/VAE_clim_clim_conv_T_tend_R_2_glob_mean',T_tend_R_2_glob_mean)
np.save('R_2_val/VAE_clim_clim_conv_Q_tend_R_2_glob_mean',Q_tend_R_2_glob_mean)    
