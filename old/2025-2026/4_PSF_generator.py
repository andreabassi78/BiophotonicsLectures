# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 23:00:29 2021
Creates a 3D PSF starting from circular pupil
@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt

um = 1.0
mm = 1000.0
pi = np.pi

Npixels = 128 # Pixels in x,y and number of planes z
n = 1 # refractive index
wavelength = 0.532*um 
f = 10*mm # focal length of the objective lens
a = 4*mm  # radius of the the pupil
k = n/wavelength # wavenumber
NA = n*np.sin(a/f) # Numerical aperture

# define the space at the pupil
b = 15 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

kx = XP * k / f 
ky = YP * k / f 

k_perpendicular = np.sqrt(kx**2 + ky**2) # k perpendicular
k_cut_off = NA/wavelength # cut off frequency in the coherent case

# create a constant ATF
ATF = np.ones([Npixels, Npixels])                  
# cut frequencies outside of the cutoff
cut_idx = (k_perpendicular >= k_cut_off)
ATF[cut_idx] = 0

ASF = ifftshift(ifft2(ATF)) # Amplitude Spread Function   
PSF = np.abs(ASF)**2 # Point Spread Function  

# calculate the space at the object plane
dxy = 1/2/np.amax(kx)
print(dxy)
x = y = np.linspace (- dxy*Npixels/2, + dxy*Npixels/2, Npixels)

fig0, ax0 = plt.subplots()
ax0.imshow(PSF, extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)])
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title(f'|PSF(x,y|')

OTF = fftshift(fft2(PSF)) # Optical Transfer Function 
MTF = np.abs(OTF) # Modulation Trnasfer Function

fig1, ax1 = plt.subplots()
ax1.plot(kx[Npixels//2,:], MTF[Npixels//2,:])
plt.xlabel('kx (1/um)')
plt.ylabel('MTF (arbitrary units)')
plt.title(f'|MTF(kx,ky|')

plt.show()