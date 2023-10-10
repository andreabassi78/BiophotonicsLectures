# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 23:00:29 2021

Creates a 3D PSF starting from circul pupil

@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt

um = 1.0
mm = 1000.0

Npixels = 256 # Pixels in x,y and number of planes z

n = 1 # refractive index

wavelength = 0.532*um 

f = 10*mm # focal length of the objective lens

a = 8*mm  # radius of the the pupil

k = n/wavelength # wavenumber

NA = n*a/f # Numerial aperture (assuming Abbe sine condition)

dz = 0.05 * um

# %% Start calculation

# define the space at the pupil
b = 50 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

kx = XP * k / f 
ky = YP * k / f 

k_rho = np.sqrt(kx**2 + ky**2) # k perpendicular
kz = np.sqrt(k**2-k_rho**2)

k_cut_off = NA/wavelength # cut off frequency in the coherent case

# create a constant ATF
ATF0 = np.ones( [Npixels, Npixels], np.complex)
cut_idx = (k_rho >= k_cut_off) # indexes of the locations outside of the pupil
ATF0[cut_idx] = 0

# insert an internal mask 
# k_cut_in = k_cut_off *0.95
# cut_idx = (k_rho <= k_cut_in) # indexes of the locations inside a certain radius
# ATF0[cut_idx] = 0
                   
PSF3D = np.zeros(((Npixels,Npixels,Npixels)))

# calculate the space at the object plane
dr = 1/2/np.amax(kx)
#x = y = dr * (np.arange(Npixels) - Npixels // 2)
x = y = np.linspace (- dr*Npixels/2, + dr*Npixels/2, Npixels)

#z = dz * (np.arange(Npixels) - Npixels // 2)
z = np.linspace (- dz*Npixels/2, + dz*Npixels/2, Npixels)


idx = 0
   
for zi in z:
   
    angular_spectrum_propagator = np.exp(1.j*2*np.pi*kz*zi)
    
    ATF = ATF0 * angular_spectrum_propagator
    mask_idx = (k_rho >= k)
    ATF[mask_idx] = 0 # Cut frequency larger than the cutoff to avoid nan error

    ASF = ifftshift(ifft2(fftshift(ATF))) # Amplitude Spread Function
    
    PSF = np.abs(ASF)**2 # Point Spread Function
    
    OTF = fftshift(fft2(ifftshift(PSF)))
    
    PSF3D[idx,:,:] = PSF
    
    idx += 1
 
       
# %% draw figure
plane_y = round(Npixels/2)
plane_z = round(Npixels/2)

fig2, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=False)
axs[0].set_title('|PSF(x,y,0)|')  
axs[0].set(xlabel = 'x ($\mu$m)')
axs[0].set(ylabel = 'y ($\mu$m)')
axs[0].imshow(PSF3D[plane_z,:,:], extent = [np.amin(x)+dr,np.amax(x),np.amin(y)+dr,np.amax(y)])


axs[1].set_title('|PSF(x,0,z)|')  
axs[1].set(xlabel = 'x ($\mu$m)')
axs[1].set(ylabel = 'z ($\mu$m)')
axs[1].imshow(PSF3D[:,plane_y,:], extent = [np.amin(x)+dr,np.amax(x),np.amin(z),np.amax(z)])
