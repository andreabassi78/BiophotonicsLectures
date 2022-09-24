# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 22:10:16 2021

Simulates, with a Poisson distribution, the fluorescence signal of single emitter (molecule) 
detected at the camera of a microsccope

@author: Andrea Bassi
"""

import numpy as np
import matplotlib.pyplot as plt

def printf(string, *values):
    for value in values:
        string += f' {value:.3f}'    
    print(string)

um = 1
pi = np.pi
delta = 2*um

Iexc = 20 # excitation intensity (a.u.) 
SNR = 5 # signal (mean) to background (std) ratio

sigma = 0.3*um # PSF size

X0 = 0.1*um # molecule position
Y0 = 0.2*um

a = 0.15*um # pixel size at the object

x = y = np.arange(-delta/2, delta/2, a)
X, Y = np.meshgrid(x,y)
area = delta**2

psf = np.exp((-(X-X0)**2-(Y-Y0)**2)/2/sigma**2)
signal = np.random.poisson(Iexc*psf) 

bm = 0 # background mean
bs = np.mean(signal)/SNR
background = np.random.normal(bm,bs,X.shape).astype('int32')
b = np.std(background)

detected = signal + background

N = np.sum(detected[:])
print('Number of photons:', N )

centroidX = np.sum(X[:]*detected[:])/np.sum(detected[:])
centroidY = np.sum(Y[:]*detected[:])/np.sum(detected[:])
varX = np.sum((X[:]-centroidX)**2*detected[:])/np.sum(detected[:])
varY = np.sum((Y[:]-centroidY)**2*detected[:])/np.sum(detected[:])
stdXY = np.sqrt((varX+varY)/2)

printf('Centroid:', centroidX, centroidY)
printf('Centroid std:', stdXY)

# accuracy = np.sqrt( sigma**2/N+a**2/12/N + 8*pi*sigma**4*b**2/a**2/N**2 ) 
accuracy = np.sqrt( sigma**2/N ) 
printf('Theoretical accuracy:', accuracy)

error = np.sqrt( (centroidX-X0)**2 + (centroidY-Y0)**2 )
printf('Localization error:  ', error)

plt.imshow(detected, 
           interpolation = None,
           extent = [-delta/2, delta/2,
                     -delta/2, delta/2]
           )
plt.plot(centroidX,-centroidY,'o')
plt.set_cmap('gray')
plt.colorbar()