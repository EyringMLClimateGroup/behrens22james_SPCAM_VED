"""
  compute the WK diagram by S. Wang.

"""


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


def dispersion_relation(k,rlat,he,wtype):
    pi    = np.pi;     re    = 6.37122e06;     g     = 9.80665;     omega = 7.292e-05;
    beta  = 2.*omega*np.cos(rlat*pi/180)/re;
    sqrt = np.sqrt
    w = np.zeros(k.shape)
    
    for i in np.arange(k.shape[0]):
        kn  = 2.*pi*k[i]/1000;
        # anti-symmetric curves
        if wtype==1 :              # MRG wave
            if kn<0.:
                dels  = sqrt(1.+(4.*beta)/(kn**2*sqrt(g*he)));
                deif = kn*sqrt(g*he)*(0.5-0.5*dels);
            if kn==0.:
                deif = sqrt(sqrt(g*he)*beta);
            if kn>0.:
                deif = np.nan

        if wtype==2:               # n=0 IG wave
            if kn<0.:
                deif = np.nan          
            if (kn==0.):
                deif = sqrt(sqrt(g*he)*beta);

            if (kn>0.):
                deli  = sqrt(1.+(4.0*beta)/(kn**2*sqrt(g*he)));
                deif = kn*np.sqrt(g*he)*(0.5+0.5*deli);
         
        if wtype==3:               # n=2 IG wave
            n=2.;
            del1  = (beta*sqrt(g*he));
            deif = sqrt((2.*n+1.)*del1 + (g*he)*kn**2);
            # do some corrections to the above calculated f==ency.......
            for j in np.arange(1,6): 
                deif = sqrt((2.*n+1.)*del1 + (g*he)*kn**2 + g*he*beta*kn/deif);

        # symmetric curves
        if wtype==4:            # n=1 ER wave
            n=1.;
            if (kn<0.):
                del1  = (beta/sqrt(g*he))*(2.*n+1.);
                deif = -beta*kn/(kn**2 + del1)
            else:
                deif = np.nan
    
        if wtype==5:               # Kelvin wave
            if (kn > 0):
                deif = kn*np.sqrt(g*he);
            else:
                deif = np.NaN
        if wtype==6:              # n=1 IG wave
            n=1.;
            del1  = (beta*sqrt(g*he));
            deif = sqrt((2.*n+1.)*del1 + (g*he)*kn**2);
            # do some corrections to the above calculated f==ency.......
            for j in np.arange(1,6):
                deif = sqrt((2.*n+1.)*del1 + (g*he)*kn**2 + g*he*beta*kn/deif);
          
        w[i]= deif*3600.0/(2.*pi); # frequency output, unit: hour^{-1}

    return w

 
def smth5(A,npass=1):
    """
      a simple 5 point smoother
      B = smth5(A,npass)
      npass: number of smoothing pass, npass = 1 by default    
    """
    nr, nc = A.shape
    B = copy.deepcopy(A)
    C = copy.deepcopy(A)
    
    for ipass in np.arange(1,npass+1):
        B[1:nr-1,1:nc-1] = (C[0:nr-2,1:nc-1] + C[2:nr,1:nc-1] + C[1:nr-1,0:nc-2] + C[1:nr-1,2:nc] + 4*C[1:nr-1,1:nc-1])*0.125;
        C=copy.deepcopy(B)
    return B


def  smth121(A,npass = 1):
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
    # see diagnostics_cam.ncl, wkSpaceTime
    nx,ny,nt = rain.shape
    print(nx,ny,nt)

    spd=1;    #sample per day
    print(spd)
    pi=np.pi
 
    rlat=0.0;
    hres=(lon[1]-lon[0])*(2*pi*6400)/360*np.cos(rlat*pi/180); # [km] zonal gridpoint spacing 27.75

    nSampTot = nt
    nSampWin = spd*96;  nSampSkip = -nSampWin/3;
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
        plt.title('Background')
        
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
        plt.title('Symmetric/Background')
        
        
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



