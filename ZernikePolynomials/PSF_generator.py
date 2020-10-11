# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:02:07 2020

Creates an abberrated wavefront in the pupil of a lens and calculates the corresponding 2D PSF

@author: Andrea Bassi
"""

import numpy as np
from zernike_polynomials import nm_polynomial
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

um = 1.0

mm = 1000*um

n = 1 # refractive index

wavelength = 0.532*um 

f = 10*mm # focal length of the objective lens

a = 5*mm  # radius of the the pupil

k = n/wavelength # wavenumber

alpha = np.arctan(a/f) # collection angle of the lens
NA = n*np.sin(alpha) # non paraxial NA
#NA = n*a/f #in paraxial approximation

print('The numerical aperture of the system is:', NA) 
print('The (Rayleigh) resolution is:', 1.22*wavelength/2/NA ,'um') 

k_cut_off = NA/wavelength # cut off frequency in the coherent case

Npixels = 128
b = 15*mm # let's define a spatial extent of the pupil, larger than the pupil radius
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

# kx = XP * k / f # in paraxial approximation 
# ky = YP * k / f # in paraxial approximation 
kx = XP / a * k_cut_off 
ky = YP / a * k_cut_off

# (kx,ky) in radial coordinates
k_rho = np.sqrt(kx**2 + ky**2)
k_theta = np.arctan2(ky,kx)

N = 3 # Zernike radial order 
M = 1 # Zernike azimutal frequency

phase = nm_polynomial(N, M, k_rho/k_cut_off, k_theta, normalized = False) 

weight = 3.14

ATF = np.exp (1.j * weight * phase) # Amplitude Transfer Function

ATF = ATF * (k_rho <= k_cut_off) # Creates a circular mask

ASF = ifftshift(ifft2(ATF)) #* k**2/f**2 # Amplitude Spread Function

PSF = np.abs(ASF)**2 # Point Spread Function

# %%% plot the ATF and the PSF 
fig, ax = plt.subplots(1, 2, figsize=(9, 4), tight_layout=False)
fig.suptitle(f'Wavefront aberrated with Zernike coefficient ({N},{M})')

im0=ax[0].imshow(np.angle(ATF), 
                 #cmap='gray',
                 extent = [np.amin(kx),np.amax(kx),np.amin(kx),np.amax(kx)],
                 )
ax[0].set_xlabel('kx (1/$\mu$m)')
ax[0].set_ylabel('ky (1/$\mu$m)')
ax[0].set_title('ATF (phase)')
fig.colorbar(im0,ax = ax[0])



x = np.fft.fftfreq(Npixels, kx[0,1]-kx[0,0])
y = np.fft.fftfreq(Npixels, ky[1,0]-ky[0,0])

extent = [min(x),max(x), min(y),max(y)]
im1=ax[1].imshow(PSF, 
                 #cmap='hot',
                 extent = extent
                 )
ax[1].xaxis.zoom(4) 
ax[1].yaxis.zoom(4)
ax[1].set_xlabel('x ($\mu$m)')
ax[1].set_ylabel('y ($\mu$m)')
ax[1].set_title('PSF')
  



