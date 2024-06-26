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

wavelength = 0.520*um 

f = 10*mm # focal length of the objective lens

a = 0.5*mm  # radius of the the pupil

k = n/wavelength # wavenumber

alpha = np.arctan(a/f) # collection angle of the lens

NA = n*np.sin(alpha) # non paraxial NA
#NA = n*a/f #in paraxial approximation

print('The numerical aperture of the system is:', NA) 
print('The diffraction limited (Rayleigh) resolution is:', 1.22*wavelength/2/NA ,'um') 

k_cut_off = NA/wavelength # cut off frequency in the coherent case

Npixels = 128
b = 1 * mm # let's define a spatial extent of the pupil, larger than the pupil radius
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

# kx = XP * k / f # in paraxial approximation 
# ky = YP * k / f # in paraxial approximation 
kx = XP / a * k_cut_off 
ky = YP / a * k_cut_off

# (kx,ky) in radial coordinates
k_rho = np.sqrt(kx**2 + ky**2)
k_theta = np.arctan2(ky,kx)

N = 0 # Zernike radial order 
M = 0 # Zernike azimutal frequency

phase = np.pi* nm_polynomial(N, M, k_rho/k_cut_off, k_theta, normalized = False) 

weight = 1

ATF = np.exp (1.j * weight * phase) # Amplitude Transfer Function

z = 200*um # defocus distance
kz =np.sqrt(k**2-k_rho**2)
angular_spectrum_propagator = np.exp(1.j*2*np.pi*kz*z)

#evanescent_idx = (k_rho >= k)
#evanescent_idx = np.isnan(angular_spectum_propagator)
#angular_spectum_propagator[evanescent_idx] = 0 # exclude all kx,ky that would be evanescent

ATF = ATF * angular_spectrum_propagator

mask_idx = (k_rho > k_cut_off)
ATF[mask_idx] = 0 # Creates a circular mask

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
ax[1].xaxis.zoom(1) 
ax[1].yaxis.zoom(1)
ax[1].set_xlabel('x ($\mu$m)')
ax[1].set_ylabel('y ($\mu$m)')
ax[1].set_title('PSF')
  



