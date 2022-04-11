"""
This script computes the MSE and biases of reference ANN (Rasp et al., 2018) for the test, validation and training data set

 To compare MSEs and biases with the VAE, AE and LR the predicted / ground truth 
SP variables were inversely normalised using the native output scaling dictionary of Rasp et al. 2018
, which yields absolute values of dT/dt in K/s and other variables in the resp. units.
As a second step these absolute values in heating, moistening tendencies are transformed using the 
VAE output normalization dictionary to compute the MSEs and biases with respect to this scaling.   
"""
from tensorflow.keras.layers import Lambda, Input, Dense
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


reference_ANN=load_model('saved_models/reference_ANN/model.h5') ## load the ANN of Rasp et al. 2018

reference_ANN.summary() # show architecture of reference ANN
# load the VAE output loss dictionary 
scale_array=ps.read_csv('nn_config/scale_dicts/Scaling_cond_VAE.csv')


PHQ_std_surf=scale_array.PHQ_std.values[-1]

TPHYSTND_std_23=scale_array.TPHYSTND_std.values[-1]

PRECT_std=scale_array.PRECT_std.values
FSNS_std=scale_array.FSNS_std.values
FSNT_std=scale_array.FSNT_std.values
FLNS_std=scale_array.FLNS_std.values
FLNT_std=scale_array.FLNT_std.values



# resulting VAE output scaling
scale_dict_II = {
    'PHQ': 1/PHQ_std_surf, 
    'TPHYSTND': 1/TPHYSTND_std_23, 
    'FSNT': 1/FSNT_std, 
    'FSNS': 1/FSNS_std, 
    'FLNT': 1/FLNT_std, 
    'FLNS': 1/FLNS_std, 
    'PRECT': 1/PRECT_std
}


import pickle
## load output normalization dictionary of Rasp et al., 2018
scale_dict_pnas= pickle.load(open('nn_config/scale_dicts/002_pnas_scaling.pkl','rb'))

in_vars = ['QBP', 'TBP','VBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']
# VBP= meridional wind component used in Rasp et al., 2018


## test data with VAE normalization dictionary
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



## test data with Rasp normalization 
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

### Test data 
print('start Test evaluation')

time_max=4415
X_1=np.nan*np.zeros((time_max,8192,94))
Y_1=np.nan*np.zeros((time_max,8192,65))
Y_1_emul=np.nan*np.zeros((time_max,8192,65))

for i in np.arange(X_1[:,1,1].size):
    X_1[i],Y_1[i]=val_gen_I[i]
    Y_1_emul[i]=reference_ANN.predict(X_1[i]) ## predicting with Rasp normalization
    
Y_1_1=val_gen_I.output_transform.inverse_transform(np.reshape(Y_1,(time_max*8192,65)))
# inverse transformation to get absolute values in respective units 
Y_emul=val_gen_I.output_transform.inverse_transform(np.reshape(Y_1_emul,(time_max*8192,65)))


mse=np.mean((val_gen_II.output_transform.transform(Y_emul[:,0:65])-val_gen_II.output_transform.transform(Y_1_1[:,0:65]))**2)
# transforming absolute values with VAE output normalization dictionary --> get reference ANN performance in VAE scaling 
print(mse)

mean_bias=np.mean(val_gen_II.output_transform.transform(Y_emul[:,0:65])-val_gen_II.output_transform.transform(Y_1_1[:,0:65]))
print(mean_bias)

del Y_1, Y_emul, X_1

### validation data set

print('start Validation evaluation')

## in Rasp scaling
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
## in VAE scaling
val_gen_IV = DataGenerator(
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
val_X=np.nan*np.zeros((time_max,8192,94))
val_Y=np.nan*np.zeros((time_max,8192,65))
val_Y_emul=np.nan*np.zeros((time_max,8192,65))


for i in np.arange(val_X[:,1,1].size):
    val_X[i],val_Y[i]=val_gen_III[i]
    val_Y_emul[i]=reference_ANN.predict(val_X[i])
    print(i)

val_Y_=val_gen_III.output_transform.inverse_transform(np.reshape(val_Y,(time_max*8192,65)))
val_Y_emul_=val_gen_III.output_transform.inverse_transform(np.reshape(val_Y_emul,(time_max*8192,65)))

val_mse=np.mean((val_gen_IV.output_transform.transform(val_Y_emul_[:,0:65])-val_gen_IV.output_transform.transform(val_Y_[:,0:65]))**2)
print(val_mse)

val_mean_bias=np.mean(val_gen_IV.output_transform.transform(val_Y_emul_[:,0:65])-val_gen_IV.output_transform.transform(val_Y_[:,0:65]))
print(val_mean_bias)
del val_Y, val_Y_emul, val_X

## training data set 
print('start Training evaluation')

## in Rasp scaling
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


## in VAE scaling 

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


train_X=np.nan*np.zeros((time_max,8192,94))
train_Y=np.nan*np.zeros((time_max,8192,65))
train_Y_emul=np.nan*np.zeros((time_max,8192,65))


for i in np.arange(train_X[:,1,1].size):
    train_X[i],train_Y[i]=train_gen_III[i]
    train_Y_emul[i]=reference_ANN.predict(train_X[i])
    print(i)

train_Y_=train_gen_III.output_transform.inverse_transform(np.reshape(train_Y,(time_max*8192,65)))
train_Y_emul_=train_gen_III.output_transform.inverse_transform(np.reshape(train_Y_emul,(time_max*8192,65)))

train_mse=np.mean((train_gen_IV.output_transform.transform(train_Y_emul_[:,0:65])-train_gen_IV.output_transform.transform(train_Y_[:,0:65]))**2)
print(train_mse)

train_mean_bias=np.mean(train_gen_IV.output_transform.transform(train_Y_emul_[:,0:65])-train_gen_IV.output_transform.transform(train_Y_[:,0:65]))
print(train_mean_bias)

# saving of MSEs and biases  

Perf_array=ps.DataFrame({'test_mse':[mse],
                            'test_bias':[mean_bias],
                            'val_mse':[val_mse],
                            'val_bias':[val_mean_bias],
                            'train_mse':[train_mse],
                            'train_bias':[train_mean_bias]})

Perf_array.to_csv('perf_analy/reference_ANN.csv')
