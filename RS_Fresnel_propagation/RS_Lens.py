"""
Created on Wed Aug 27 16:43:01 2020

Propagates a field throught a lens using Fresnel integral and considering the lens 
as a quadratic phase mask

@author: Andrea Bassi
"""

import numpy as np
from numpy import pi, exp, sqrt
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt

def kernelFresnel(X, Y, z, wavelength, n):
    """Kernel for Fesnel propagation

    Parameters:
        X (numpy.array): positions x
        Y (numpy.array): positions y
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refraction index of background

    Returns:
        complex np.array: kernel
    """  
    
    k =  n / wavelength
    return - 1.j*k/z * exp( 1.j*2*pi*k*z + 1.j*pi*k/z*(X**2 + Y**2) ) 

def show_fields( fields, titles, kind = 'intensity', extent = (-50,50,-50,50) ):
    """ Shows fields with matplotlib """
    
    _fig, axs = plt.subplots(1,len(fields))
    
    for idx, E in enumerate(fields):
        
        if kind == 'real':
            data_to_show  = np.real(E) # Real part
        elif kind == 'phase':
            data_to_show  = np.angle(E) # Phase
        elif kind == 'intensity':     
            data_to_show  = np.abs(E)**2 # Intensity
        else:   
            data_to_show  = np.abs(E) # Magnitude
        
        axs[idx].imshow(data_to_show, interpolation='none',
                        cmap='gray',
                        origin='lower',
                        extent = extent,
                        vmin = 0,
                        vmax = None
                        )
        ylabel = 'y ($\mu$m)' if idx== 0 else None
        axs[idx].set(xlabel = 'x ($\mu$m)',
                     ylabel = ylabel,
                     title = titles[idx],
                     ) 

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
D1 = kernelFresnel(X, Y, z1, wavelength, n)

# %%calculate E1_minus, the field just before the lens
E1_minus = ifftshift( ifft2 (fft2(E0) * fft2(D1) ) ) * dx *dy 

# %%calculate E1_plus, the field just after the lens
f = 300 *um
lens = np.exp(- 1.j * 2 * pi * k * ((X**2 / (2 * f)) + Y**2 / (2 * f)))
E1_plus = E1_minus * lens 

# %% calculate the second free space propagator
D2 = kernelFresnel(X, Y, z2, wavelength, n)

# %% calculate E2, the field at z1+z2  
E2 = ifftshift( ifft2 (fft2(E1_plus) * fft2(D2) ) ) * dx *dy 

#%% show the fields as intensity, phase or real part
show_fields(fields = (E0,E1_minus,E2),
            titles = ('E0','E1-','E2'),
            kind = 'abs'
            )