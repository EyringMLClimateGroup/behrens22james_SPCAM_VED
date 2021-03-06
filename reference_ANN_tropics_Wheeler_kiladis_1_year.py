"""
This script computes the Wheeler-Kiladis diagram based on 1 year of
OLR predictions of reference ANN (Rasp et al.,2018) and SP data 
"""
from tensorflow.keras.layers import Lambda, Input, Dense
from cbrain.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, Callback


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib as mpl

import tensorflow as tf
from cbrain.imports import *

from cbrain.utils import *
import pandas as ps
from cbrain.data_generator import DataGenerator

reference_ANN=load_model('saved_models/reference_ANN/model.h5') ## load the ANN of Rasp et al. 2018
import pickle
# load the native output normalization of Rasp et al., 2018
scale_dict_pnas= pickle.load(open('nn_config/scale_dicts/002_pnas_scaling.pkl','rb'))

in_vars = ['QBP', 'TBP','VBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS', 'PRECT']
## use the entire second year of SPCAM simulation
val_gen_I = DataGenerator(
    data_fn = '../preprocessed_data/005_valid_1_year.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '../preprocessed_data/001_norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_pnas,
    batch_size=8192,
    shuffle=True
)

lat=np.arange(-90,90,180/64)
lon=np.arange(-180,180,360/128)
# The number of time steps was calculated before this analysis based on the length of the used
# data array divided by the number of samples per time step (lat.size*lon.size=8192) 
time=np.arange(17519)
latit_array=np.reshape((lat.T*np.ones((lat.size,lon.size)).T).T,int(lat.size*lon.size))


latit_timestep_array=np.reshape((latit_array.T*np.ones((latit_array.size,time.size)).T),int(latit_array.size*time.size))

## select tropics between 15°S and 15°N
tropics=np.where((latit_timestep_array<15)&(latit_timestep_array>-15))[0]
tropic=np.where((latit_array<15)&(latit_array>-15))[0]
trops=np.where((lat<15)&(lat>-15))[0]

OLR_array=np.nan*np.zeros((time.size,int(lat[trops].size*lon.size)))
# select OLR data set from SPCAM simulations (OLR output var ID [62])
print('Start OLR sampling')
for i in np.arange(time.size):
    OLR_array[i]=val_gen_I.output_transform.inverse_transform(val_gen_I[i][1])[tropic,62]
    
    
print('End OLR sampling')


input_reference_ANN=np.nan*np.zeros((time.size,int(lat[trops].size*lon.size),94))
print('Start input sampling')

for i in np.arange(time.size):
    input_reference_ANN[i]=(val_gen_I[i][0])[tropic,:]
    
print('End input sampling')
    
# reshape SP OLR data into (time, lat(trops), lon)     
OLR_array_lat_lon=np.reshape(OLR_array,(time.size,lat[trops].size,lon.size))

input_reference_ANN_conc=np.reshape(input_reference_ANN,(time.size*lat[trops].size*lon.size,94))
# predict OLR with reference ANN 
print('start NN prediction')
reference_ANN_OLR=val_gen_I.output_transform.inverse_transform(reference_ANN.predict(input_reference_ANN_conc))[:,62]
print('End NN prediction')
# reshape reference ANN OLR data set into (time,lat(trops),lon)    
reference_ANN_OLR_array_lat_lon=np.reshape(reference_ANN_OLR,(time.size,lat[trops].size,lon.size))

def load_olr_and_highpass_filter(olr_data,lat_,lon_,filter=True, bandpass=[1/96,1/2]):
    """
    fft analysis in time 
    
    author: Shuguang Wang <sw2526@columbia.edu>
    """
    
    nlat=lat_.size
    nlon=lon_.size
    f_sample =1/(1800)# time step of SPCAM data (1800s)
    
   
    olr_obs_filt  = np.zeros(olr_data.shape)
    fft_freq = np.fft.fftfreq(olr_data.shape[0])*f_sample*3600*24
    
    print(fft_freq)
    for ii in np.arange(nlon):
        for jj in np.arange(nlat):
 
            otmp = olr_data[:,jj,ii]            
            fft_coef = np.fft.fft(otmp)
            if filter==True:
                ifreq_sel = np.where((np.abs(fft_freq)<=bandpass[0]))[0]
                print(ifreq_sel)
                fft_coef[ifreq_sel] = 0.0 
                ifreq_sel_min = np.where((np.abs(fft_freq)>bandpass[1]))[0]
                print(ifreq_sel_min)
                fft_coef[ifreq_sel_min] = 0.0 
            
            otmp = np.real(np.fft.ifft(fft_coef))
            olr_obs_filt[:,jj,ii] = otmp

    return olr_obs_filt ,fft_freq


#apply fft in time on SP OLR data set without bandpass filter 
A_with_out_filter,fft_freq_with_out_filter=load_olr_and_highpass_filter(OLR_array_lat_lon,lat[trops],
                                                                        lon,False,bandpass=[1/30,1/2])

#apply fft in time on reference OLR data set without bandpass filter 

B_with_out,fft_freq_VAE_with_out=load_olr_and_highpass_filter(reference_ANN_OLR_array_lat_lon,lat[trops],lon,False,bandpass=[1/30,1/2])

from wk_spectrum.module_wk_diagram import dispersion_relation, calc_wk_diagram

import numpy as np
import glob
from datetime import datetime, timedelta
import re
import scipy.io
from netCDF4 import Dataset
import sys
import time
import netCDF4
import copy
import matplotlib.pyplot as plt



def smth5(A,npass=1):
    """
      a simple 5 point smoother
      B = smth5(A,npass)
      npass: number of smoothing pass, npass = 1 by default  
      
      author: Shuguang Wang <sw2526@columbia.edu>

    """
    nr, nc = A.shape
    B = copy.deepcopy(A)
    C = copy.deepcopy(A)
    
    for ipass in np.arange(1,npass+1):
        B[1:nr-1,1:nc-1] = (C[0:nr-2,1:nc-1] + C[2:nr,1:nc-1] + C[1:nr-1,0:nc-2] + C[1:nr-1,2:nc] + 4*C[1:nr-1,1:nc-1])*0.125;
        C=copy.deepcopy(B)
    return B


def  smth121(A,npass = 1):
    """
    author: Shuguang Wang <sw2526@columbia.edu>

    """
    ns = A.shape[0]
    #print(ns, npass)
    B = copy.deepcopy(A)    
    C = copy.deepcopy(A)    
    for ipass in np.arange(1,npass+1):
        B[0] = (3*C[0] + C[1])*0.25;
        B[1:ns-1] = (C[0:ns-2] + C[2:ns]+2*C[1:ns-1])*0.25;
        B[ns-1] = (3*C[ns-1] + C[ns-2])*0.25;
        C = B*1.0
        #print(B)
        #print(ipass)
    return B



def calc_wk_diagram(rain, lon, title='', nsmth = 10, plot=True):
    """
    calculate Wheeler-Kiladis diagram
    """
    # see diagnostics_cam.ncl, wkSpaceTime
    nx,ny,nt = rain.shape
    print(nx,ny,nt)

    spd=48;    #sample per day
    print(spd)
    pi=np.pi
 
    rlat=0.0;
    hres=(lon[1]-lon[0])*(2*pi*6400)/360*np.cos(rlat*pi/180); # [km] zonal gridpoint spacing of CAM grid

    nSampTot = nt
    nSampWin = spd*60;# increase sample window to 60 days for 1 year of SPCAM simulations (for 3month =30 days)
    nSampSkip = -nSampWin/3;
    print(nSampWin)
    nWindow   = (nSampTot*1.0-nSampWin)/(nSampWin+nSampSkip)  + 1;
    nWindow
    hlon = int(np.ceil((nx+1.0)/2));
    ht = int(np.ceil((nSampWin+1.0)/2))
    #decompose to asym,sym

    ny2 = int(ny/2)+1
    rains = np.zeros((nx,ny2,nt))
    raina = np.zeros((nx,ny2,nt))
    for j in np.arange(ny2) : 
        rains[:,j,:]=0.5*(rain[:,j,:]+rain[:,ny-j-1,:])    # Symmetric component
        raina[:,j,:]=0.5*(rain[:,j,:]-rain[:,ny-j-1,:]);  # antiSymmetric component
    rains[:,ny2-1,:] = rain[:,ny2-1,:]
    raina[:,ny2-1,:] = 0.0
    print(ny, ny2)
    pwrwin_s = np.zeros((nx,nSampWin))
    pwrwin_a = pwrwin_s*0.0

    rain2d = np.zeros((nx,nSampWin))
    for iwin in np.arange(1, int(np.ceil(nWindow))):
        pwr_s = np.zeros((nx,nSampWin))
        pwr_a = np.zeros((nx,nSampWin))
        for j in np.arange(ny2):  # loop through all latitude
            tstart = int((iwin-1)*(nSampWin+nSampSkip))
            tend = tstart + nSampWin        
            rain2d_tmp = rains[:,j,tstart:tend]
            fftrain2d  = np.fft.fft2(rain2d_tmp)/(nx*nSampWin);
            pwr_s = pwr_s + np.abs(fftrain2d[:,:])**2     
            
            rain2d_tmp = raina[:,j,tstart:tend]
            fftrain2d  = np.fft.fft2(rain2d_tmp)/(nx*nSampWin);
            pwr_a = pwr_a + np.abs(fftrain2d[:,:])**2     
        pwrwin_s = pwrwin_s + pwr_s/int(np.ceil(nWindow))*2
        pwrwin_a = pwrwin_a + pwr_a/int(np.ceil(nWindow))*2
        
        
    print(fftrain2d.shape)
    pwrshift = np.fft.fftshift(pwrwin_s);
    pwrplot_s = np.zeros((nx, ht))+ np.nan
    pwrplot_s[:, 0:ht]= np.fliplr(pwrshift[:,0:ht])

    pwrshift = np.fft.fftshift(pwrwin_a);
    pwrplot_a = np.zeros((nx, ht)) + np.nan
    pwrplot_a[:, 0:ht]= np.fliplr(pwrshift[:,0:ht])
    
    x1 = np.arange(-(nx-1.0)/2, (nx-1)/2+1)/(nx*hres) # [1/km] zonal wavenumber
    x1 = np.arange(-(nx-1.0)/2, (nx-1)/2+1) # zonal wavenumber

    y1 = np.arange(0, nSampWin/2.0+1)/(nSampWin/spd); # [1/hr] frequency
    #print(y1)
    [x,y]=np.meshgrid(x1,y1);

    # now we make background spectrum by loop through all latitude
    pwrsmth_alllat = np.zeros((nx,nSampWin))
    for iwin in np.arange(1, int(np.ceil(nWindow))):
        pwr = np.zeros((nx,nSampWin))
        for j in np.arange(ny2):  # loop through all latitude
            tstart = int((iwin-1)*(nSampWin+nSampSkip))
            tend = tstart + nSampWin        
            rain2d[:,:] = rains[:,j,tstart:tend]
            fftrain2d_1  = np.fft.fft2(rain2d[:,:])/(nx*nSampWin);
            rain2d[:,:] = raina[:,j,tstart:tend]
            fftrain2d_2  = np.fft.fft2(rain2d[:,:])/(nx*nSampWin);            
            pwr[:,:] = pwr[:,:] + np.abs(fftrain2d_1 )**2 + np.abs(fftrain2d_2 )**2       
        pwrsmth_alllat = pwrsmth_alllat + pwr/int(np.ceil(nWindow))
    pwrshift = np.fft.fftshift(pwrsmth_alllat);
    pwrplot_sum = np.zeros((nx, ht))
    pwrplot_sum[:, 0:ht]= np.fliplr(pwrshift[:,0:ht])
    pwrplot_sum[:,0] = np.nan
     
    
    pwrsmth = np.zeros((nx,ht)); 
    maxavesmth = 27
    if 1 == 1:
        for j in np.arange(nx):  # smooth over frequency
            pwrplot_s[j,1:ht] = smth121(pwrplot_s[j,1:ht],1)
            pwrplot_a[j,1:ht] = smth121(pwrplot_a[j,1:ht],1)
        
    for i in np.arange(1,ht):  # smooth over wavenumber
        if y1[i] < 0.1:
            Msmth = 5*2
        elif y1[i] >= 0.1 and  y1[i] < 0.2:
            Msmth = 10*2
        elif y1[i] >= 0.2 and  y1[i] < 0.3:
            Msmth = 20
        elif y1[i] >= 0.3 :
            Msmth = 40    
        pwrsmth[maxavesmth:-maxavesmth,i] = smth121(pwrplot_sum[maxavesmth:-maxavesmth,i], Msmth)
    for j in np.arange(nx):  # smooth over frequency
        pwrsmth[j,1:ht] = smth121(pwrsmth[j,1:ht],nsmth)


    pwrdiv_s = np.zeros((nx,ht)) + np.nan
    pwrdiv_a = np.zeros((nx,ht)) + np.nan        
    pwrdiv_s[:,1:ht] = pwrplot_s[:,1:ht]/pwrsmth[:,1:ht];
    pwrdiv_a[:,1:ht] = pwrplot_a[:,1:ht]/pwrsmth[:,1:ht];
       
    
    wlimit=0.5 # [1/day]
    klimit=10
   
    if plot == True: 
        x11 = np.arange(-(nx-1)/2, (nx-1)/2+0.1, 0.1)/(nx*hres); # [1/km] zonal wavenumber
        x112 = np.arange(-(nx-1)/2, (nx-1)/2+0.1, 0.1); # [1/km] zonal wavenumber
    
        plt.figure(figsize=(10, 10))
        plt.subplot(3,2,1)
        plt.contourf(x, y, np.log10(pwrplot_s).T,  15,  cmap='jet', extend='both')
        plt.colorbar()
        plt.axis([-klimit, klimit, 0, wlimit])
        plt.title('log10(raw), Symmetric')
    
        plt.subplot(3,2,2)
        plt.contourf(x, y, np.log10(pwrplot_a).T,  15,  cmap='jet', extend='both')
        plt.colorbar()
        plt.axis([-klimit, klimit, 0, wlimit])
        plt.title('log10(raw), Asymmetric')
        
        plt.subplot(3,2,3)
        plt.contourf(x, y, np.log10(pwrsmth).T,  15,  cmap='jet', extend='both')
        plt.colorbar()
        plt.axis([-klimit, klimit, 0, wlimit])
        plt.title('Background')
        
        
        plt.subplot(3,2,5)
        plt.contourf(x, y, pwrdiv_s.T, levels=np.arange(0.2, 2.1, 0.1), vmin=0.1, vmax=2.1, cmap='jet', extend='both')
        plt.colorbar()
        plt.axis([-klimit, klimit, 0, wlimit])
        plt.title('Symmetric/Background')
    
        plt.subplot(3,2,6)
        plt.contourf(x, y, pwrdiv_a.T, levels=np.arange(0.5,1.6, 0.1), cmap='jet', extend='both')
        plt.colorbar()
        plt.axis([-klimit, klimit, 0, wlimit])
        plt.title('Asymmetric/Background')
        
        
        for i in np.arange(6):
            if i == 3:
                continue
            plt.subplot(3,2,i+1)
            ax = plt.gca()
            if i in [0,4]:
                for w in np.arange(4,7): #asym 1:3, sym 4:6
                    ax.plot(x112,dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
                    ax.plot(x112,dispersion_relation(x11,rlat,50,w)*24, 'k', linewidth=0.5)
                    ax.plot(x112,-dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
                    ax.plot(x112,-dispersion_relation(x11,rlat,50,w)*24, 'k', linewidth=0.5)
                #plt.plot(x112,dispersion_relation(x11,rlat,4,5)*24)
            elif i in [1,5]:
                for w in [1,2,3]: #asym 1:3, sym 4:6
                    ax.plot(x112,dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
                    ax.plot(x112,dispersion_relation(x11,rlat,50,w)*24, 'k', linewidth=0.5)
                    ax.plot(x112,-dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
                    ax.plot(x112,-dispersion_relation(x11,rlat,50,w)*24, 'k', linewidth=0.5)
                #plt.plot(x112,dispersion_relation(x11,rlat,4,5)*24)
                
            ax.axis([-klimit, klimit, 0, wlimit])
            
            plt.xlabel('Wavenumber')
            plt.ylabel('Frequency (1/d)')
            #plt.title(title, y=1.03)
        
        plt.tight_layout()
    
    return x1, y1, pwrdiv_s, pwrdiv_a, pwrplot_s, pwrplot_a, pwrsmth


# calculate Wheeler-Kiladis diagram for 1 year SP OLR data set
x_wn_with_out, y_freq_with_out, pwrdiv_sym_with_out, pwrdiv_anti_with_out, pwrplot_sym_with_out, pwrplot_anti_with_out, pwrsmth_olr_obs_with_out = calc_wk_diagram(np.transpose(A_with_out_filter,[2,1,0]), lon , title='OLR', plot=False )

# calculate Wheeler-Kiladis diagram for 1 year of reference ANN OLR predictions 
x_wn_reference_ANN_with_out, y_freq_reference_ANN_with_out, pwrdiv_sym_reference_ANN_with_out, pwrdiv_anti_reference_ANN_with_out, pwrplot_sym_reference_ANN_with_out, pwrplot_anti_VAE_with_out, pwrsmth_olr_obs_VAE_with_out = calc_wk_diagram(np.transpose(B_with_out,[2,1,0]), lon , title='OLR', plot=False )


# load colormap for Wheeler-Kiladis diagram of .ncl script 
import wk_spectrum.nlcmap
from wk_spectrum.colors import gen_cmap
cmap_amwg_blueyellowred = gen_cmap('amwg_blueyellowred')


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_waves(lon, ax, opt='sym'):
    """
    plot characteristic dispersion relations of tropical waves
    
    author: Shuguang Wang <sw2526@columbia.edu>
    """
    
    nx = len(lon)    
    rlat = 0.0
    hres=(lon[1]-lon[0])*(2*np.pi*6400)/360*np.cos(rlat*np.pi/180); # [km] zonal gridpoint spacing 27.75
    
    x11 = np.arange(-(nx-1)/2, (nx-1)/2+0.1, 0.1)/(nx*hres) # [1/km] zonal wavenumber
    x112 = np.arange(-(nx-1)/2, (nx-1)/2+0.1, 0.1); # [1/km] zonal wavenumber
    
    if opt == 'sym':
        for w in [4,5,6]: #asym 1:3, sym 4:6
            ax.plot(x112,dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
            ax.plot(x112,dispersion_relation(x11,rlat,90,w)*24, 'k', linewidth=0.5)
            ax.plot(x112,-dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
            ax.plot(x112,-dispersion_relation(x11,rlat,90,w)*24, 'k', linewidth=0.5)
        #plt.plot(x112,dispersion_relation(x11,rlat,4,5)*24)
    elif opt == 'anti':
         for w in [1,2,3]: #asym 1:3, sym 4:6
            ax.plot(x112,dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
            ax.plot(x112,dispersion_relation(x11,rlat,90,w)*24, 'k', linewidth=0.5)
            ax.plot(x112,-dispersion_relation(x11,rlat,12,w)*24, 'k', linewidth=0.5)
            ax.plot(x112,-dispersion_relation(x11,rlat,90,w)*24, 'k', linewidth=0.5)        
        
    klimit = 15
    wlimit = 0.5
    ax.axis([-klimit, klimit, 0, wlimit])   

#plot_waves(olons, plt.gca())    


cn_int = np.hstack((np.arange(0.2,1.3,0.1), [1.4, 1.7, 2, 2.4, 2.8, 3]))
cmap_test = ListedColormap(cmap_amwg_blueyellowred)
cmap_nonlin = nlcmap.nlcmap(cmap_test, cn_int)



x_wn_15_15=np.where((x_wn_reference_ANN_with_out<=15)&(x_wn_reference_ANN_with_out>=-15))[0][:]
y_freq_0_0_5=np.where((y_freq_reference_ANN_with_out<=0.5)&(y_freq_reference_ANN_with_out>=0))[0]

x_wn_15_15_0=np.ones((1,1))*np.array(np.where((x_wn_reference_ANN_with_out<=15)&(x_wn_reference_ANN_with_out>=-15))[0][:])
#print(y_freq_0_0_5)
x_wn_15_15_1=x_wn_15_15_0.astype(int)
y_freq_0_0_5_0=np.ones((1,1))*np.array(np.where((y_freq_reference_ANN_with_out<=0.5)&(y_freq_reference_ANN_with_out>=0))[0])
y_freq_0_0_5_1=y_freq_0_0_5_0.astype(int)

# compute difference between W-K diagrams of reference ANN OLR and SP OLR
# within -15 < k zonal wave number < 15 and 0 < frequency < 0.5 days^-1 

SP_reference_ANN_disp_diff=(-pwrdiv_sym_with_out[x_wn_15_15_1[:],y_freq_0_0_5_1[:].T]+pwrdiv_sym_reference_ANN_with_out[x_wn_15_15_1,y_freq_0_0_5_1.T])

## plot Wheeler-Kiladis diagram of SP OLR and reference ANN OLR for 1 year of data 
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.contourf(x_wn_with_out, y_freq_with_out, pwrdiv_sym_with_out.T, levels=cn_int,  cmap=cmap_nonlin,  extend='both' )
plt.xlim([-15, 15])
plt.ylim([0, 0.5])
plt.colorbar()
plt.contour(x_wn_with_out, y_freq_with_out, pwrdiv_sym_with_out.T, levels=cn_int,  colors='k', linewidths=0.2 )
x0 = np.arange(0,0.51,0.05)
plt.plot(x0*0, x0, '--', color='k', linewidth=0.7)
plt.xlabel('Wavenumber')
plt.ylabel(r'Frequency $[\frac{1}{day}]$')
plot_waves(lon, plt.gca())
plt.text(-15, -0.05, 'Westward')
plt.text(10, -0.05, 'Eastward')

plt.title(r'Q$_{lw \ top}$ SP',Fontsize=16)
 
plt.tight_layout()


plt.subplot(1,2,2)
plt.contourf(x_wn_reference_ANN_with_out, y_freq_reference_ANN_with_out, pwrdiv_sym_reference_ANN_with_out.T, levels=cn_int,  cmap=cmap_nonlin,  extend='both' )
plt.xlim([-15, 15])
plt.ylim([0, 0.5])
plt.colorbar()
plt.contour(x_wn_reference_ANN_with_out, y_freq_reference_ANN_with_out, pwrdiv_sym_reference_ANN_with_out.T, levels=cn_int,  colors='k', linewidths=0.2 )
x0 = np.arange(0,0.51,0.05)
plt.plot(x0*0, x0, '--', color='k', linewidth=0.7)
plt.xlabel('Wavenumber',Fontsize=16)
plt.ylabel(r'Frequency $[\frac{1}{day}]$',Fontsize=16)
plot_waves(lon, plt.gca())
plt.text(-15, -0.05, 'Westward',Fontsize=16)
plt.text(10, -0.05, 'Eastward',Fontsize=16)
plt.title(r'Q$_{lw \ top}$ reference ANN',Fontsize=20)
 
plt.tight_layout()

plt.savefig('wheeler_kiladis/reference_ANN_OLR_1_year_15NS_fixed_1_2.png')

## plot difference reference ANN - SP 
plt.figure(5,(5,5))
plt.contourf(x_wn_reference_ANN_with_out[x_wn_15_15], y_freq_reference_ANN_with_out[y_freq_0_0_5],SP_reference_ANN_disp_diff, levels=41, cmap=plt.cm.seismic,vmin=-0.5,vmax=0.5, extend='both')
plt.xlim([-15, 15])
plt.ylim([0, 0.5])
x0 = np.arange(0,0.51,0.05)
plt.plot(x0*0, x0, '--', color='k', linewidth=0.7)
plt.xlabel('Wavenumber',Fontsize=16)
plt.ylabel(r'Frequency $[\frac{1}{day}]$',Fontsize=16)
plot_waves(lon, plt.gca())
plt.text(-15, -0.05, 'Westward',Fontsize=16)
plt.text(10, -0.05, 'Eastward',Fontsize=16)
plt.title(r' difference Q$_{lw \ top}$ reference ANN - SP',Fontsize=20)
bx, _ = mpl.colorbar.make_axes(plt.gca())

a=mpl.cm.ScalarMappable(cmap=plt.cm.seismic, norm=mpl.colors.Normalize(vmin=-0.5, vmax=0.5))
a.set_clim([-0.5, 0.5])                      
plt.colorbar(a,cax=bx)
 

plt.savefig('wheeler_kiladis/reference_ANN_OLR_1_year_15NS_diff_fixed_1.png')

### plot W-K diagrams of SP OLR, reference ANN OLR and difference reference ANN - SP 

plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.contourf(x_wn_with_out, y_freq_with_out, pwrdiv_sym_with_out.T, levels=cn_int,  cmap=cmap_nonlin,  extend='both' )
plt.xlim([-15, 15])
plt.ylim([0, 0.5])
#plt.colorbar()
plt.contour(x_wn_with_out, y_freq_with_out, pwrdiv_sym_with_out.T, levels=cn_int,  colors='k', linewidths=0.2 )
x0 = np.arange(0,0.51,0.05)
plt.plot(x0*0, x0, '--', color='k', linewidth=0.7)
#plt.xlabel('Wavenumber',Fontsize=20)
plt.ylabel(r'$\omega$ $[\frac{1}{day}]$',Fontsize=24)
plt.xticks(Fontsize=16)
plt.yticks(Fontsize=16)

plot_waves(lon, plt.gca())
plt.text(-15, -0.06, 'Westward',Fontsize=20)
plt.text(10, -0.06, 'Eastward',Fontsize=20)
plt.title(r'a) Outgoing Longwave Rad. Q$_{lw \ top}$ SP',Fontsize=24)
 
plt.tight_layout()


plt.subplot(1,3,2)
plt.contourf(x_wn_reference_ANN_with_out, y_freq_reference_ANN_with_out, pwrdiv_sym_reference_ANN_with_out.T, levels=cn_int,  cmap=cmap_nonlin,  extend='both' )
plt.xlim([-15, 15])
plt.ylim([0, 0.5])
cb_1=plt.colorbar()
cb_1.ax.tick_params(labelsize=16)
plt.contour(x_wn_reference_ANN_with_out, y_freq_reference_ANN_with_out, pwrdiv_sym_reference_ANN_with_out.T, levels=cn_int,  colors='k', linewidths=0.2 )
x0 = np.arange(0,0.51,0.05)
plt.plot(x0*0, x0, '--', color='k', linewidth=0.7)
plt.xlabel(r'k',Fontsize=24)
#plt.ylabel('Frequency (/day)',Fontsize=20)
plot_waves(lon, plt.gca())
plt.xticks(Fontsize=16)
plt.yticks(Fontsize=16)

#plt.text(-15, -0.05, 'Westward',Fontsize=20)
#plt.text(10, -0.05, 'Eastward',Fontsize=20)
plt.title(r'b) Q$_{lw \ top}$ reference ANN',Fontsize=24)
 
plt.tight_layout()


plt.subplot(1,3,3)
plt.contourf(x_wn_reference_ANN_with_out[x_wn_15_15], y_freq_reference_ANN_with_out[y_freq_0_0_5],SP_reference_ANN_disp_diff, levels=41, cmap=plt.cm.seismic,vmin=-0.5,vmax=0.5, extend='both')
plt.xlim([-15, 15])
plt.ylim([0, 0.5])
x0 = np.arange(0,0.51,0.05)
plt.plot(x0*0, x0, '--', color='k', linewidth=0.7)
#plt.xlabel('Wavenumber',Fontsize=20)
#plt.ylabel('Frequency (/day)',Fontsize=20)
plot_waves(lon, plt.gca())
plt.xticks(Fontsize=16)
plt.yticks(Fontsize=16)

plt.text(-15, -0.06, 'Westward',Fontsize=20)
plt.text(10, -0.06, 'Eastward',Fontsize=20)
plt.title(r'c) Q$_{lw \ top}$ reference ANN - SP',Fontsize=24)
bx, _ = mpl.colorbar.make_axes(plt.gca())
#cbar = mpl.colorbar.ColorbarBase(ax, cmap=plt.cm.seismic,
#                       norm=mpl.colors.Normalize(vmin=-1, vmax=1))
a=mpl.cm.ScalarMappable(cmap=plt.cm.seismic, norm=mpl.colors.Normalize(vmin=-0.5, vmax=0.5))
a.set_clim([-0.5, 0.5])                      
cb_2=plt.colorbar(a,cax=bx)
 
cb_2.ax.tick_params(labelsize=16)





plt.savefig('wheeler_kiladis/reference_ANN_OLR_1_year_15NS_diff_fixed_combo_1.png')






