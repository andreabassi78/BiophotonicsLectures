import numpy as np 
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift,ifftshift
from scipy.signal import convolve
import os
import matplotlib.image as mpimg

full_path = os.path.realpath(__file__) 
folder, _ = os.path.split(full_path) # this selects the folder of this .py file  
sample_filename = 'filaments.tif'
psf_filename = 'psf_aberrated.tif'

sample = np.float32(mpimg.imread(folder+'/'+sample_filename))
psf = np.float32(mpimg.imread(folder+'/'+psf_filename))

sample = sample/np.amax(sample)
psf = psf/np.sum(psf)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(sample,cmap='gray')
plt.title('Original sample')
plt.subplot(1, 2, 2)
plt.imshow(psf,cmap='gray')
plt.title('Point Spread Function')


F_sample = fftshift(fft2(sample))
F_psf = fftshift(fft2(psf)) # Optical Transfer funtion

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.abs(F_sample))
plt.title('Sample spectrum')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(F_psf))
plt.title('Optical Transfer Function')

image = np.abs(ifftshift(ifft2(F_sample*F_psf)))
#image = convolve(sample, psf, 'same')

alpha = 0.05
noise = np.random.rand(sample.shape[0], sample.shape[1])
image += alpha* noise

plt.figure()
plt.imshow(image,cmap='gray')
plt.colorbar()
plt.title('Image at the detector')
plt.show()
