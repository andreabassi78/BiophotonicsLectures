'''
Created on 2 ago 2019
Shows examples of spatial filtering a 2D image in Fourier domain
@author: Andrea Bassi, Politecnico di Milano
Lecture on 2D Fourier Transforms and filtering 
Optical Microscopy Course (Biophotonics)

'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from ImageType import image
from Function2D import function2D

#####################create ImageType with squares
testimage=image('file')   #choose between 'file' and 'rect'  
Z=testimage.im
KX, KY = testimage.createKspace()

#####################################################################################
# calculate 2D fft (translated to the center using fftshift. ifftshift is used to remove phase error)
ft=fftshift(fft2(ifftshift(Z)))

filter_function = function2D(KX, KY)
kmax=np.amax(KX)
filter_function.functiontype('square', kmax/20) 
"""
choose between 'circle', 'annulus', 'square', 'gaussian', 'ramp', 'delta'
"""

filt = filter_function.data 

#filter ImageType in Fourier space
ft_filtered= ft*filt

##################################################################################
# create filtered ImageType
Z_filtered=ifftshift(ifft2(fftshift(ft_filtered)))

# create filter impulse response
SpreadFunction=ifftshift(ifft2(fftshift(filt)))

epsilon = 1e-9 #constant to avoid log 0 later

# create figures
fig = plt.figure(figsize=(16, 9))
plt.title('Transfer Function: ' + filter_function.title + '\n\n')
plt.axis('off')

ax=[]
# ax enables access to manipulate each of subplots

# create subplot:
ax.append(fig.add_subplot(2, 3, 1))
ax[0].set_title("Original image")  # set title
plt.imshow(np.abs(Z), interpolation='none', cmap='gray' )

# create subplot:
ax.append(fig.add_subplot(2, 3, 4))
ax[1].set_title("Original image Power spectral density (log)")  # set title
psd = 10*np.log10( abs(ft+epsilon)**2)
plt.imshow(psd, interpolation='none' , cmap='gray')
#Note that if the ImageType Z is real, FT is Hermitian --> the abs and the real are radially simmetric. The imag has opposite sign

# create subplot:
ax.append(fig.add_subplot(2, 3, 2))
ax[2].set_title("Impulse response")  # set title
plt.imshow(np.abs(SpreadFunction), interpolation='none', cmap='gray')

# create subplot:
ax.append(fig.add_subplot(2, 3, 5))
ax[3].set_title("Transfer function, Power spectral density (log)")  # set title
psd = 10*np.log10( abs(filt+epsilon)**2)
plt.imshow(psd, interpolation='none' , cmap='gray')

# create subplot:
ax.append(fig.add_subplot(2, 3, 3))
ax[4].set_title("Filtered image")  # set title
plt.imshow(np.abs(Z_filtered), interpolation='none', cmap='gray')

# create subplot:
ax.append(fig.add_subplot(2, 3, 6))
ax[5].set_title("Filtered image Power spectral density (log)")  # set title
psd = 10*np.log10( abs(ft_filtered+epsilon)**2)
plt.imshow(psd, interpolation='none', cmap='gray')

# finally, render the plots
plt.show()
