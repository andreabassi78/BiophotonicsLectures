"""
Created on Wed May 20 23:21:44 2020

Based on the kernel functions found in Diffractio module (https://pypi.org/project/diffractio/)

Notation from Mertz, "Introduction to Optical Microscopy"

Simulates the propagation of a field using Rayleight-Sommerfield or Fresnel integrals 

@author: Andrea Bassi
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp
from numpy.fft import fft2,ifft2, ifftshift


def kernelRS(X, Y, z, wavelength, n):
    """Kernel for Rayleight-Sommerfield propagation

    Parameters:
        X (numpy.array): positions x
        Y (numpy.array): positions y
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refraction index of background
        kind (str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel    

    """
    
    k = n / wavelength
    r = sqrt(X**2 + Y**2 + z**2)
    return - 1.j*k*z/r * exp(1.j * 2*pi*k*r)/r * ( 1 - 1 /(1.j*2*pi*k*r) )
    
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
wavelength = 0.532 * um 
n = 1


Nsamples = 1024 # number of pixels
L = 300 * um # extent of the xy space
x = y = np.linspace(-L/2, +L/2, Nsamples)
X, Y = np.meshgrid(x,y)

""" create a constant field E0 (plane wave propagating along z)"""
E0 = np.ones([Nsamples, Nsamples])

# """ create a plane wave with certain kx and ky """  
# kx = 0.0
# ky = 0.0
# E0 = np.exp(-1.j*2*pi* (kx*X+ky*Y))
# #E0 = np.cos(2*pi* (kx*X+ky*Y)) 

""" insert a square mask """
side = 10 * um
indexes = (np.abs(X)>side/2) | (np.abs(Y)>side/2)
E0[indexes] = 0

"""calculate the free space propagator """
z = 1000 * um
D = kernelRS(X, Y, z, wavelength, n)
#D = kernelFresnel(X, Y, z, wavelength, n)

""" calculate E1 as the convolution of E0 and D, using Fast Fourier Transform """
dx = x[1]-x[0]
dy = y[1]-y[0]
E1 = ifftshift( ifft2 (fft2(E0) * fft2(D) ) ) * dx *dy 

""" show the fields as magnitude, intensity, phase or real part """
show_fields(fields = (E0,E1),
            titles = ('E0', f'E1, z= {z}'),
            kind = 'abs',
            extent = (-L/2,L/2,-L/2,L/2)
            )