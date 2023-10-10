# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 23:00:29 2021

Generates a 2D PSF and OTF starting from a circular pupil
and plots the MTF

@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt

um = 1.0
mm = 1000.0
Npixels = 200 # Pixels in x,y

n = 1 # refractive index
wavelength = 0.532*um 
f = 20*mm # focal length of the objective lens
a = 10*mm  # radius of the the pupil
k = n/wavelength # wavenumber
NA = n*a/f # Numerial aperture (assuming small angles)

# create the pupil
b = 50 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)
rhoP = np.sqrt(XP**2+YP**2) 
pupil = np.ones([Npixels, Npixels])
#pupil[xP >0] = -1 # STED-like 
pupil[rhoP>a] = 0
# pupil[rhoP<a/2] = 0 # Bessel-like

plt.figure()
plt.imshow(pupil)
plt.colorbar()

ASF = ifftshift(ifft2(pupil)) # Amplitude Spread Function
PSF = np.abs(ASF)**2 # Point Spread Function
PSF = PSF

plt.figure()
plt.imshow(PSF)

OTF = fftshift(fft2(PSF))
MTF = np.abs(OTF)

plt.figure()
plt.imshow(MTF)
plt.colorbar()

plt.figure()
kx = xP * k / f 
ky = yP * k / f 
plt.plot(kx, MTF[Npixels//2,:])
plt.xlabel('kx (1/um)')
plt.ylabel('MTF (a.u.)')

plt.show()