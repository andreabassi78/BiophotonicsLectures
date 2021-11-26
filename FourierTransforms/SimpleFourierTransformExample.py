# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:09:04 2020

@author: Andrea Bassi
"""


import numpy as np
import matplotlib.pyplot as plt

L = 1 # extent of the space to be considered
N = 512 # number of pixels
x = y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x,y)


# %% Create a gaussian function
# sigma = 0.01
# Z = np.exp(-(X**2+Y**2)/sigma**2)


# %% Create a rectangle function
Z = np.zeros([N,N])
side = 0.01
indexes = (np.abs(X)<side/2) & (np.abs(Y)<side/2)
Z[indexes] = 1


# %% Show the function
plt.figure()
plt.imshow(Z)
plt.gray()
plt.title('original image')

#%% Calculate and show the Fourier Transform
Zk = np.fft.fftshift(np.fft.fft2(Z))
plt.figure()
plt.imshow(np.abs(Zk))
plt.title('Fourier transform')
