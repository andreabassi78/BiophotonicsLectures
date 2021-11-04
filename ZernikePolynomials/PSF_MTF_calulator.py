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

a = 4*mm  # radius of the the pupil

k = n/wavelength # wavenumber

NA = n*a/f # Numerial aperture (assuming Abbe sine condition)

print('The numerical aperture of the system is:', NA) 
print('The diffraction limited (Rayleigh) resolution is:', 1.22*wavelength/2/NA ,'um') 

k_cut_off = NA/wavelength # cut off frequency in the coherent case

Npixels = 256
b = 15 * mm # let's define a spatial extent of the pupil, larger than the pupil radius
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

kx = XP * k / f 
ky = YP * k / f 
# kx = XP / a * k_cut_off # other way to calculate kx and ky
# ky = YP / a * k_cut_off

# (kx,ky) in radial coordinates
k_rho = np.sqrt(kx**2 + ky**2)
k_theta = np.arctan2(ky,kx)

N = 2 # Zernike radial order 
M = 0 # Zernike azimutal frequency

phase = nm_polynomial(N, M, k_rho/k_cut_off, k_theta, normalized = False) 

weight = 0.25 # weight of the polynomials in units of lambda (weight 1 means  wavefront abberated of lamba/2)

ATF = np.exp (1.j * 2* np.pi* weight * phase) # Amplitude Transfer Function

mask_idx = (k_rho > k_cut_off)
ATF[mask_idx] = 0 # Creates a circular mask

ASF = ifftshift(ifft2(fftshift(ATF))) # Amplitude Spread Function

PSF = np.abs(ASF)**2 # Point Spread Function
PSF = PSF/np.sum(PSF) # PSF normalized with its area

OTF = fftshift(fft2(ifftshift(PSF)))

MTF = np.abs(OTF) # Modulation Transfer Function
# change to real to see the negative values of the MTF

# %%% plot the ATF and the PSF 
fig = plt.figure(figsize=(9, 9))
fig.suptitle(f'Wavefront aberrated with Zernike coefficient ({N},{M})')


ax0 = plt.subplot(221)
im0=ax0.imshow(np.angle(ATF), 
                 #cmap='gray',
                 extent = [np.amin(kx),np.amax(kx),np.amin(kx),np.amax(kx)],
                 )
ax0.set_xlabel('kx (1/$\mu$m)')
ax0.set_ylabel('ky (1/$\mu$m)')
ax0.set_title('ATF (phase)')
fig.colorbar(im0,ax = ax0)

ax1 = plt.subplot(222)
x = np.fft.fftfreq(Npixels, kx[0,1]-kx[0,0])
y = np.fft.fftfreq(Npixels, ky[1,0]-ky[0,0])

extent = [min(x),max(x), min(y),max(y)]
im1=ax1.imshow(PSF, 
                 #cmap='hot',
                 extent = extent
                 )
ax1.xaxis.zoom(4) 
ax1.yaxis.zoom(4)
ax1.set_xlabel('x ($\mu$m)')
ax1.set_ylabel('y ($\mu$m)')
ax1.set_title('PSF')
 

ax2 = plt.subplot(212) 
cx = int(MTF.shape[1]/2)
plt1=ax2.plot(kx[cx,:],MTF[cx,:])
ax2.set_xlabel('kx (1/$\mu$m)')
ax2.set_ylabel('MTF')
ax2.grid(True)

pupil_phase = (np.angle(ATF[k_rho < k_cut_off]))/(2*np.pi)
print('The peak to valley (P-V) wavefront error is:', 
      np.amax(pupil_phase)- np.amin(pupil_phase)
      ) 

print('The root mean sqared (RMS) wavefront error is:', 
      np.sqrt(np.mean(pupil_phase**2))
      ) 
