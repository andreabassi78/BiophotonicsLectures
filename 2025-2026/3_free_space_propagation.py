import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp
from numpy.fft import fft2, ifft2, ifftshift
                        
um = 1.0
wavelength = 0.532 * um 
n = 1

Nsamples = 1024 # number of pixels
L = 300 * um # extent of the xy space
x = y = np.linspace(-L/2, +L/2, Nsamples)
X, Y = np.meshgrid(x,y)

""" create a constant field E0 (plane wave propagating along z)"""
E0 = np.ones([Nsamples, Nsamples])

""" insert a square mask """
side = 30 * um
indexes = (np.abs(X)>side/2) | (np.abs(Y)>side/2)
E0[indexes] = 0

