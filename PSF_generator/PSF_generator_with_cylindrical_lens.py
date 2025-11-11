# -*- coding: utf-8 -*-
"""
Calculate a PSF out of focus with added astigmatism
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
NA = n*(a/f) # Numerical aperture

# define the space at the pupil
b = 15 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

kx = XP * k / f # assuming Abbe's sine condition
ky = YP * k / f # assuming Abbe's sine condition

k_perpendicular = np.sqrt(kx**2 + ky**2) # k perpendicular
k_cut_off = NA/wavelength # cut off frequency in the coherent case

# create a constant ATF
ATF = np.ones([Npixels, Npixels])                  

# add astigmatism 
#R_pupil_eff    = 3.0e-3      # [m] effective entrance pupil radius at the cylinder
f_cyl_m        = 1.0         # [m] focal length of the cylindrical lens

rho   = kx / k_cut_off
theta = np.arctan2(ky, kx)

xprime = rho
phi_cyl = (2.0*np.pi / wavelength) * (xprime**2) / (2.0 * f_cyl_m)

ATF = ATF * np.exp(1j * phi_cyl)

# add defocus
z = 0*um
kz = np.sqrt(k**2-k_perpendicular**2)
angular_spectrum_propagator = np.exp(1.j*2*pi*kz*z)
ATF = ATF * angular_spectrum_propagator

# cut frequencies outside of the pupil
cut_idx = (k_perpendicular >= k_cut_off) # indexes of the locations outside of the pupil 
ATF[cut_idx] = 0

ASF = ifftshift(ifft2(ATF)) # Amplitude Spread Function   
PSF = np.abs(ASF)**2 # Point Spread Function  

# calculate the space at the object plane
dr = 1/2/np.amax(kx)
x = y = np.linspace (- dr*Npixels/2, + dr*Npixels/2, Npixels)

fig0, ax0 = plt.subplots()
ax0.imshow(PSF, extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)])
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title(f'|PSF(x,y,z={z}um)|')

OTF = fftshift(fft2(PSF)) # Optical Transfer Function 
MTF = np.abs(OTF) # Modulation Trnasfer Function

fig1, ax1 = plt.subplots()
ax1.plot(kx[Npixels//2,:], MTF[Npixels//2,:])
plt.xlabel('kx (1/um)')
plt.ylabel('MTF (arbitrary units)')
plt.title(f'|MTF(kx,ky,z={z}um)|')

plt.show()