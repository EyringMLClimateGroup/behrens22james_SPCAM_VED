"""
training of the VAE_climate_convection network

"""

# import some packages 
from tensorflow.keras.layers import Lambda, Input, Dense
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

# reparameterization trick
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
original_dim_output=65  # output node size (SP variables)



# network parameters
input_shape = (original_dim_input,)
out_shape=(original_dim_output,)
intermediate_dim = 463 # node size of first hidden layers of encoder 
batch_size = 714
latent_dim = 5 # latent space width = 5 nodes
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
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Posteriour model
first_layer_width=353 # fixed node size of hidden layers of ANN coupled to encoder 
SPCAM_output_num=65 # output variables / number of SP variables 
post_net_inp_lay=Input((latent_dim,),name='posterior_net_input')
x_0 =Dense(first_layer_width, activation='relu')(post_net_inp_lay)
x_1 =Dense(first_layer_width, activation='relu')(x_0)
x_2 =Dense(first_layer_width, activation='relu')(x_1)

post_outputs = Dense(SPCAM_output_num, activation='elu')(x_2)
poster_network=Model(post_net_inp_lay,post_outputs,name='posterior')
poster_network.summary()

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
encoder_loss = K.mean(kl_loss*weight)
#instantiate VAE model
emul_outputs=poster_network(encoder(inputs)[2])
coupled_net=Model(inputs,emul_outputs)
coupled_net.add_loss(encoder_loss)
coupled_net.add_metric(kl_loss, name='kl_loss', aggregation='mean')




# loading the scaling dictionary for the output normalization
scale_array=ps.read_csv('nn_config/scale_dicts/Scaling_cond_VAE.csv')


PHQ_std_surf=scale_array.PHQ_std.values[-1]

TPHYSTND_std_23=scale_array.TPHYSTND_std.values[-1] # use std on 845 hPa for dT/dt 

PRECT_std=scale_array.PRECT_std.values
FSNS_std=scale_array.FSNS_std.values
FSNT_std=scale_array.FSNT_std.values
FLNS_std=scale_array.FLNS_std.values
FLNT_std=scale_array.FLNT_std.values


# resulting output scaling dictionary 
scale_dict_II = {
    'PHQ': 1/PHQ_std_surf, 
    'TPHYSTND': 1/TPHYSTND_std_23, 
    'FSNT': 1/FSNT_std, 
    'FSNS': 1/FSNS_std, 
    'FLNT': 1/FLNT_std, 
    'FLNS': 1/FLNS_std, 
    'PRECT': 1/PRECT_std
}

in_vars = ['QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']

# Takes representative value for PS since purpose is normalization
PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];






from cbrain.data_generator import DataGenerator

#training data (shuffled in space and time) = July, August and September of first year of SPCAM simulation
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


#validation data = April, May and June of second year

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
# Learning rate schedule
def schedule(epoch):
    
  if epoch < 7:
    return 0.00018238
  elif epoch < 14:
    return 0.00018238/5

  elif epoch < 21:
    return 0.00018238/25

  elif epoch < 28:
     
    return 0.00018238/125

  elif epoch < 35:
        
    return 0.00018238/625

  else:
    return 0.00018238/3125
    
# load encoder states from previous VAE experiment optimized for predicting CAM variables  
encoder.load_weights('./saved_models/VAE_climate_encoding/encoder_JAS.h5')
coupled_net.compile(tf.keras.optimizers.Adam(lr=1E-4), loss=mse, metrics=['mse'])
callback_lr=LearningRateScheduler(schedule,verbose=1)
coupled_net.fit((train_gen),validation_data=(val_gen,None),epochs=40,shuffle=False,callbacks=
                              [callback_lr,AnnealingCallback(weight)])


coupled_net.save_weights('saved_models/VAE_climate_convection/My_Model_JAS_40_1_anneal_opt_net.h5')
coupled_net.save_weights('saved_models/VAE_climate_convection/My_Model_JAS_40_1_anneal_opt_net.tf')

hist_df = ps.DataFrame(coupled_net.history.history) 

hist_csv_file = 'saved_models/VAE_climate_convection/My_Model_JAS_40_1_anneal_opt_net_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


