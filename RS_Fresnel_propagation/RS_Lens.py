"""
Created on Wed Aug 27 16:43:01 2020

Propagates a field throught a lens using RS integral and considering the lens as a quadratic phase mask

@author: Andrea Bassi
"""

import numpy as np
from numpy import pi, exp
from numpy.fft import fft2, ifft2, ifftshift
from RS_Fresnel import kernelRS, show_fields

um = 1.0
      
n = 1
wavelength = 0.532 * um 
z1 = 300 * um
z2 = 300 * um 

k = n / wavelength

Nsamples = 1024 # number of pixels
L = 100 * um # extent of the xy space
x = y = np.linspace(-L/2, +L/2, Nsamples)
X, Y = np.meshgrid(x,y)
dx = x[1]-x[0]
dy = y[1]-y[0]

# %% create a constant field E0 and a mask
E0 = np.ones([Nsamples, Nsamples])
side = 30 * um
indexes = (np.abs(X)>side/2) | (np.abs(Y)>side/2)
E0[indexes] = 0

# %% calculate the first free space propagator
D1 = kernelRS(X, Y, z1, wavelength, n)

# %%calculate E1_minus, the field just before the lens
E1_minus = ifftshift( ifft2 (fft2(E0) * fft2(D1) ) ) * dx *dy 

# %%calculate E1_plus, the field just after the lens
f = 300 *um
lens = np.exp(- 1.j * 2 * pi * k * ((X**2 / (2 * f)) + Y**2 / (2 * f)))
E1_plus = E1_minus * lens 

# %% calculate the second free space propagator
D2 = kernelRS(X, Y, z2, wavelength, n)

# %% calculate E2, the field at z1+z2  
E2 = ifftshift( ifft2 (fft2(E1_plus) * fft2(D2) ) ) * dx *dy 

#%% show the fields as intensity, phase or real part
show_fields(fields = (E0,E1_minus,E2),
            titles = ('E0','E1-','E2'),
            kind = 'abs'
            )