"""
train the conditional VAE network
"""

# import some packages 
from tensorflow.keras.layers import Lambda, Input, Dense, Concatenate 
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

# kl annealing for increase in reproduction skill of cVAE 
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


# the initial value of weight is 0
# define it as a keras backend variable
weight = K.variable(0.)

    
original_dim_input=int(65+64)  # cVAE is trained on SP and CAM variables 

original_dim_output=65 # and reproduces SP variables 


# network parameters

intermediate_dim = 457 # number of nodes in first / last hidden layers of Encoder / Decoder 
batch_size = 666
latent_dim = 5 # latent space width of signals passed trough encoder 
epochs = 40    
    
# network parameters
input_shape = (original_dim_input,)
out_shape=(original_dim_output,)
large_scale_fields=64    
    
    
latent_dim_cond = int(latent_dim+large_scale_fields)# node size of latent space nodes + initial CAM variables --> input to decoder

    
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

z_cond=Concatenate()([z,inputs[:,65:129]]) #here latent nodes and CAM variables are merged 

# instantiate encoder model
encoder = Model([inputs], [z_mean, z_log_var,z, z_cond], name='encoder') 
#output of encoder is mean, log-var, latent nodes z and (z+CAM) variables 
encoder.summary()


##conditional Decoder
decoder_inputs =Input(shape=(latent_dim_cond,), name='decoder_input') # has 69 input nodes due to combination of z and CAM variables 
                      
x_1 =Dense(int(np.round(intermediate_dim/16)), activation='relu')(decoder_inputs)
x_2 =Dense(int(np.round(intermediate_dim/8)), activation='relu')(x_1)
x_3 =Dense(int(np.round(intermediate_dim/4)), activation='relu')(x_2)
x_4 =Dense(int(np.round(intermediate_dim/2)), activation='relu')(x_3)
x_5 =Dense(intermediate_dim, activation='relu')(x_4)
x_6 =Dense(intermediate_dim, activation='relu')(x_5)
outputs = Dense(original_dim_output, activation='elu')(x_6)

decoder = Model([decoder_inputs], outputs, name='decoder')
decoder.summary()
emul_cond_outputs=decoder(encoder([inputs])[3])
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
VAE_loss = K.mean(kl_loss*weight)


cond_VAE=Model(inputs,emul_cond_outputs)
cond_VAE.add_loss(VAE_loss)
cond_VAE.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    
    


#loading scaling dictionary for SP variables 
scale_array_2D=ps.read_csv('nn_config/scale_dicts/Scaling_enc_II_range_profiles.csv')
scale_array_1D=ps.read_csv('nn_config/scale_dicts/Scaling_enc_II_range.csv')

TBP_std_surf=scale_array_2D.TBP_std.values[-1]

QBP_std_surf=scale_array_2D.QBP_std.values[-1]

Q_lat_std_surf=scale_array_1D.Q_lat_std.values

Q_sens_std_surf=scale_array_1D.Q_sens_std.values


Q_solar_std_surf=scale_array_1D.Q_sol_std.values

PS_std_surf=scale_array_1D.PS_std.values



# resulting scaling dictionary 
scale_dict_II = {
    'PHQ': 1/PHQ_std_surf, 
    'TPHYSTND': 1/TPHYSTND_std_23, 
    'FSNT': 1/FSNT_std, 
    'FSNS': 1/FSNS_std, 
    'FLNT': 1/FLNT_std, 
    'FLNS': 1/FLNS_std, 
    'PRECT': 1/PRECT_std
}


out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']
in_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT','QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']


# Takes representative value for PS since purpose is normalization
PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];




from cbrain.data_generator import DataGenerator

# training data (shuffled in space and time) --> July, August, September of first year of SPCAM simulations 
train_gen = DataGenerator(
    data_fn = '../preprocessed_data/1918_train_3_month_JAS_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=666,
    shuffle=True
)



# validation data April, May and June of second year  
val_gen = DataGenerator(
    data_fn = '../preprocessed_data/1918_valid_3_month_AMJ.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/000_norm_1_month.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=666,
    shuffle=True
)
# learning rate schedule
def schedule(epoch):
    
  if epoch < 7:
    return 0.0009613299673028408
  elif epoch < 14:
    return 0.0009613299673028408/5

  elif epoch < 21:
    return 0.0009613299673028408/25

  elif epoch < 28:
     
    return 0.0009613299673028408/125

  elif epoch < 35:
        
    return 0.0009613299673028408/625

  else:
    return 0.0009613299673028408/3125
    
 
cond_VAE.compile(tf.keras.optimizers.Adam(lr=1E-4), loss=mse, metrics=['mse'])
callback_lr=LearningRateScheduler(schedule,verbose=1)
cond_VAE.fit((train_gen),validation_data=(val_gen,None),epochs=40,shuffle=False,callbacks=
                              [callback_lr,AnnealingCallback(weight)])


cond_VAE.save_weights('saved_models/cVAE/New_Cond_VAE_40_opt_anneal.h5')
cond_VAE.save_weights('saved_models/cVAE/New_Cond_VAE_40_opt_anneal.tf')

hist_df = ps.DataFrame(cond_VAE.history.history) 

hist_csv_file = 'saved_models/cVAE/New_Cond_VAE_40_opt_anneal_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
