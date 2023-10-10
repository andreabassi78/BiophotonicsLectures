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

Npixels = 128 # Pixels in x,y and number of planes z
n = 1 # refractive index
wavelength = 0.532*um 
f = 10*mm # focal length of the objective lens
a = 4*mm  # radius of the the pupil
k = n/wavelength # wavenumber
NA = n*(a/f) # Numerical aperture

# define the space at the pupil
b = 20 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

kx = XP * k / f # assuming Abbe's sine condition
ky = YP * k / f # assuming Abbe's sine condition

k_perpendicular = np.sqrt(kx**2 + ky**2) # k perpendicular
k_cut_off = NA/wavelength # cut off frequency in the coherent case

# create a constant ATF
ATF = np.ones([Npixels, Npixels])                  

# add defocus
z = 0.0 * um
kz = np.sqrt(k**2-k_perpendicular**2)
angular_spectrum_propagator = np.exp(1.j*2*np.pi*kz*z)
ATF = ATF * angular_spectrum_propagator

# cut frequencies outside of the pupil
cut_idx = (k_perpendicular >= k_cut_off) # indexes of the locations outside of the pupil 
ATF[cut_idx] = 0

ASF = ifftshift(ifft2(fftshift(ATF))) # Amplitude Spread Function   
PSF = np.abs(ASF)**2 # Point Spread Function  

# calculate the space at the object plane
dr = 1/2/np.amax(kx)
x = y = np.linspace (- dr*Npixels/2, + dr*Npixels/2, Npixels)

fig0, ax0 = plt.subplots()
ax0.imshow(PSF, extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)])
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title(f'|PSF(x,y,z={z}um)|')

OTF = fftshift(fft2(ifftshift(PSF))) # Optical Transfer Function 
MTF = np.real(OTF) # Modulation Trnasfer Function

fig1, ax1 = plt.subplots()
ax1.plot(kx[Npixels//2,:], MTF[Npixels//2,:])
plt.xlabel('kx (1/um)')
plt.ylabel('MTF (arbitrary units)')
plt.title(f'|MTF(kx,ky,z={z}um)|')

plt.show()