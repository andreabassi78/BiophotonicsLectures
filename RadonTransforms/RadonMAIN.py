'''
Created on 25 ago 2019
Modified from skimage example
@author: Andrea Bassi, Politecnico di Milano
Lecture on Inverse Radon Transform and BackProjection
Optical Microscopy Course (Biophotonics)
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.io import imread

from skimage.transform import radon, iradon, rescale
from xlrd.formula import num2strg

SAMPLING_ANGLE = 2 #deg

image = imread('Shepplogan.png',True)
#image = rescale(image, scale=0.4, mode='reflect', multichannel=False)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap='gray')

theta = np.linspace(0., 180., 180./SAMPLING_ANGLE, endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap='gray',
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()

"""
Reconstruction with Filtered Back Projection
"""


t0=time.time()
reconstruction_fbp = iradon(sinogram, theta, circle=True)
print("Time for execution of FBP: " + num2strg(round(time.time()-t0,6)) +" s")
error = reconstruction_fbp-image
print(f"FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Original")
ax1.imshow(image, cmap='gray')
ax2.set_title("Reconstruction with Filtered Back Projection. \n Sampling angle = " + num2strg(SAMPLING_ANGLE) + "\xb0")
ax2.imshow(reconstruction_fbp , cmap='gray')
plt.show()


plt.show()