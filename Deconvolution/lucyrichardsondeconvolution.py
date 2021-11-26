import numpy as np 
import matplotlib.pyplot as plt

from scipy import signal
from skimage import data
from skimage import filters


camera = data.camera()
camera = np.float32(camera)
camera = camera[1:,1:]

delta = np.zeros_like(camera)

center = camera.shape[0]//2
delta[center,center]=1

psf = filters.gaussian(delta, sigma = 2)

camera_blurred = np.fft.ifft2(np.fft.fft2(camera) * np.fft.fft2(psf))
camera_blurred = np.abs(np.fft.fftshift(camera_blurred))

alpha = 10
noise = np.random.rand(camera.shape[0], camera.shape[1])

camera_blurred += alpha* noise

plt.figure()
plt.imshow(camera)
plt.title('Original')

plt.figure()
plt.imshow(camera_blurred)
plt.title('Blurred')



# %% SPECTRAL DIVISION

def spectraldivision(blurred, psf):
    F_blurred = np.fft.fft2(blurred)
    F_psf = np.fft.fft2(psf)
    F_spectraldivision = F_blurred / F_psf
    reconstructed = np.abs(np.fft.fftshift(np.fft.ifft2(F_spectraldivision)))
    return reconstructed
    
camera_deblurred = spectraldivision(camera_blurred, psf)
plt.figure()
plt.imshow(camera_deblurred)
plt.title('Spectral division')

# %% RICHARDSON LUCY ALGORITHM

def richardsonlucy(blurred, psf, iterations):
    deconvolved = blurred.copy()
    
    for i in range(iterations):
        forwardconv = signal.convolve(deconvolved, psf, 'same')
        ratio = blurred / forwardconv
        
        deconvolved = deconvolved * signal.convolve(ratio, psf[::-1, ::-1], 'same')
        # deconvolved *= signal.correlate(ratio, psf)
        
    return deconvolved
 
camera_deblurred = richardsonlucy(camera_blurred, psf, iterations=10)

plt.figure()
plt.imshow(camera_deblurred, vmax=255)
plt.title('Deconvolved with LR')