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
f = 2*mm # focal length of the objective lens
a = 1.9*mm  # radius of the the pupil
k = n/wavelength # wavenumber
NA = n*a/f # Numerial aperture (assuming Abbe sine condition)
f_cyl = 5000*mm
k_cut_off = NA/wavelength
# create the pupil
b = 5 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)
rhoP = np.sqrt(XP**2+YP**2) 

kx = XP *k /f
ky = YP *k /f
k_perp = np.sqrt(kx**2+ky**2)
kz = np.sqrt(k**2-k_perp**2)
# pupil = np.exp(-1.j*2*np.pi*k*XP**2/(2*f_cyl))
# pupil[rhoP>a] = 0
# ATF = np.exp(-1.j*2*np.pi*f**2*kx**2/(2*f_cyl)/k)

# ATF = np.exp(-1.j*2*np.pi*k*kx**2/(2*f_cyl)*f/k)
# #ATF = np.exp(-np.pi *2*f_cyl/k *kx**2/1.j)
# ATF = np.exp(-1.j*2*np.pi*k*kx**2/(2*f_cyl)*(f/k)**2)
phase = 2*np.pi*k*XP**2/(2*f_cyl)
phase = phase-np.mean(phase)
ATF = np.exp(-1.j*phase)
#ATF = np.exp(-1.j*2*np.pi*kz*0.01)#XP**2/(2*f_cyl) )
ATF[k_perp>k_cut_off] = 0

plt.figure()
plt.imshow(np.angle(ATF))
plt.colorbar()

ASF = ifftshift(ifft2(ATF)) # Amplitude Spread Function
PSF = np.abs(ASF)**2 # Point Spread Function
PSF = PSF

plt.figure()
plt.imshow(PSF)

# OTF = fftshift(fft2(PSF))
# MTF = np.abs(OTF)


# plt.figure()
# plt.imshow(MTF)
# plt.colorbar()

# plt.figure()
# kx = xP * k / f 
# ky = yP * k / f 
# plt.plot(kx, MTF[Npixels//2,:])
# plt.xlabel('kx (1/um)')
# plt.ylabel('MTF (a.u.)')

