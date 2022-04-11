"""
This script computes the R² value of dT/dt and dq/dt on 700hPa for reference ANN
Furthermore the R² values of the entire dT/dt and dq/dt profiles and the global mean R² 
profiles for dT/dt and dq/dt are calculated. 
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

import pickle
## load output normalization dictionary of Rasp et al., 2018
scale_dict_pnas= pickle.load(open('nn_config/scale_dicts/002_pnas_scaling.pkl','rb'))

in_vars = ['QBP', 'TBP','VBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']
# VBP= meridional wind component used in Rasp et al., 2018


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

time=4415
lat=np.arange(-90,90,180/64)
lon=np.arange(-180,180,360/128)
X_1=np.nan*np.zeros((time,8192,94))
Y_1=np.nan*np.zeros((time,8192,65))
Y_1_emul=np.nan*np.zeros((time,8192,65))

for i in np.arange(X_1[:,1,1].size):
    X_1[i],Y_1[i]=val_gen_I[i]
    Y_1_emul[i]=reference_ANN.predict(X_1[i])
    
Y_real=val_gen_I.output_transform.inverse_transform(np.reshape(Y_1,(4415*8192,65)))
# inverse transformation to get absolute values in respective units 
Y_emul=val_gen_I.output_transform.inverse_transform(np.reshape(Y_1_emul,(4415*8192,65)))
# inverse transformation to get absolute values in respective units 


T_tend_real=np.reshape(Y_real[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
T_tend_emul=np.reshape(Y_emul[:,30:60],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_real=np.reshape(Y_real[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))
Q_tend_emul=np.reshape(Y_emul[:,0:30],(time,lat.size,lon.size,Y_real[1,30:60].size))


## compute R² values for dT/dt and dq/dt on 700 hPa
T_tend_R_2_700=np.nan*np.zeros((lat.size,lon.size))
Q_tend_R_2_700=np.nan*np.zeros((lat.size,lon.size))

T_tend_R_2_700=1-np.mean((np.squeeze(T_tend_real[:,:,:,20])-np.squeeze(T_tend_emul[:,:,:,20]))**2,0)/np.var(np.squeeze(T_tend_real[:,:,:,20]),0)
Q_tend_R_2_700=1-np.mean((np.squeeze(Q_tend_real[:,:,:,20])-np.squeeze(Q_tend_emul[:,:,:,20]))**2,0)/np.var(np.squeeze(Q_tend_real[:,:,:,20]),0)

np.save('R_2_val/reference_ANN_T_tend_R_2_700',T_tend_R_2_700)
np.save('R_2_val/reference_ANN_Q_tend_R_2_700',Q_tend_R_2_700)


## compute R² values for entire dT/dt and dq/dt profiles 
T_tend_R_2_glob=np.nan*np.zeros((lat.size,lon.size,30))
Q_tend_R_2_glob=np.nan*np.zeros((lat.size,lon.size,30))

for z in tqdm(np.arange(T_tend_R_2_glob[1,1,:].size)):
    T_tend_R_2_glob[:,:,z]=1-np.mean((np.squeeze(T_tend_real[:,:,:,z])-np.squeeze(T_tend_emul[:,:,:,z]))**2,0)/np.var(np.squeeze(T_tend_real[:,:,:,z]),0)
    Q_tend_R_2_glob[:,:,z]=1-np.mean((np.squeeze(Q_tend_real[:,:,:,z])-np.squeeze(Q_tend_emul[:,:,:,z]))**2,0)/np.var(np.squeeze(Q_tend_real[:,:,:,z]),0)

 
## compute global mean R² values for dT/dt and dq/dt profiles based on global mean of variance and global mean sum of squared errors 
T_tend_R_2_glob_mean=np.nan*np.zeros((30))
Q_tend_R_2_glob_mean=np.nan*np.zeros((30))

for z in tqdm(np.arange(T_tend_R_2_glob_mean.size)):
    T_tend_R_2_glob_mean[z]=1-np.mean(np.mean(np.mean((np.squeeze(T_tend_real[:,:,:,z])-np.squeeze(T_tend_emul[:,:,:,z]))**2,2),1),0)/np.mean(np.mean(np.var(np.squeeze(T_tend_real[:,:,:,z]),0),1),0)
    Q_tend_R_2_glob_mean[z]=1-np.mean(np.mean(np.mean((np.squeeze(Q_tend_real[:,:,:,z])-np.squeeze(Q_tend_emul[:,:,:,z]))**2,2),1),0)/np.mean(np.mean(np.var(np.squeeze(Q_tend_real[:,:,:,z]),0),1),0)
    
np.save('R_2_val/reference_ANN_T_tend_R_2_glob',T_tend_R_2_glob)
np.save('R_2_val/reference_ANN_Q_tend_R_2_glob',Q_tend_R_2_glob)
np.save('R_2_val/reference_ANN_T_tend_R_2_glob_mean',T_tend_R_2_glob_mean)
np.save('R_2_val/reference_ANN_Q_tend_R_2_glob_mean',Q_tend_R_2_glob_mean)        