# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 22:10:16 2021

Simulates, with a Poisson distribution, the fluorescence signal of single emitter (molecule) 
detected at the camera of a microsccope

@author: Andrea Bassi
"""

import numpy as np
import matplotlib.pyplot as plt

um = 1
delta = 3*um

Iexc = 10 # excitation intensity (a.u.) 

sigma = 0.3*um # PSF size

X0 = 0.0*um # molecule position
Y0 = 0.0*um

a = 0.4*um # pixel size at the object

x = y = np.arange(-delta/2, delta/2, a)
X, Y = np.meshgrid(x,y)
area = delta**2


psf = Iexc*np.exp((-(X-X0)**2-(Y-Y0)**2)/2/sigma**2)
psf = psf.astype(np.uint32)

#background = SNR * Iexc * np.random.random(psf.shape)
b = 0
background = np.random.normal(0.0, b, psf.shape)
N = 1000
centroidX = np.zeros(N)
centroidY= np.zeros(N)
nphotons= np.zeros(N)

for i in range(N):
    detected = np.random.poisson(psf) + background
    detected = detected.astype(np.int32)
    nphotons[i] = np.sum(detected[:])
    centroidX[i] = np.sum(X[:]*detected[:])/np.sum(detected[:])
    centroidY[i] = np.sum(Y[:]*detected[:])/np.sum(detected[:])

#plt.plot(np.std(signal, axis = 0))
errorX=np.std(centroidX-X0)
errorY=np.std(centroidY-Y0)
error = np.mean(np.sqrt((centroidX-X0)**2+(centroidY-Y0)**2))
nph = np.mean(nphotons)
print('error:',error)
print('Number of photons:', nph)
#accuracy = np.sqrt( sigma**2/nph ) 
accuracy = np.sqrt( ((sigma**2+a**2/12)/nph)*(16/9+8*np.pi*(sigma**2)*(b**2)/(a**2*nph**2)) ) 


print('Theoretical accuracy:', accuracy)
print('ratio:', error/accuracy)
plt.imshow(detected, 
            interpolation = None,
            extent = [-delta/2, delta/2,
                      -delta/2, delta/2]
            )
#plt.plot(centroidX,-centroidY,'o')
plt.set_cmap('gray')
plt.colorbar()