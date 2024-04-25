import numpy as np 
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift,ifftshift
from scipy.signal import convolve
import os
import matplotlib.image as mpimg

full_path = os.path.realpath(__file__) 
folder, _ = os.path.split(full_path) # this selects the folder of this .py file  
sample_filename = 'filaments.tif'
psf_filename = 'psf.tif'

sample = np.float32(mpimg.imread(folder+'/'+sample_filename))
psf = np.float32(mpimg.imread(folder+'/'+psf_filename))

sample = sample/np.amax(sample)
psf = psf/np.sum(psf)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(sample)
plt.subplot(1, 2, 2)
plt.imshow(psf)
plt.title('Original sample and PSF')

image = convolve(sample,psf,'same')

alpha = 0.05 
noise = np.random.rand(sample.shape[0], sample.shape[1])
image += alpha* noise

plt.figure()
plt.imshow(image)
plt.colorbar()
plt.title('Image at the detector')


# %% SPECTRAL DIVISION

def spectraldivision(blurred, psf):
    F_blurred = fft2(blurred)
    F_psf = fft2(psf)
    F_spectraldivision = F_blurred / F_psf
    result = np.abs(ifftshift(ifft2(F_spectraldivision)))
    return result
    

# %% WIENER FILTER DECONVOLUTION

def wiener_filter(blurred, psf, K=0.01):

    blurred = blurred
    psf = psf/ np.sum(psf)
    blurred_fft = fft2(blurred)
    psf_fft = fft2(psf)

    # Compute Wiener filter
    psf_fft_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft)**2
    wiener_filter = psf_fft_conj / (psf_power + K)

    # Apply filter in the frequency domain
    deconvolved_fft = wiener_filter * blurred_fft

    # Inverse FFT to get the deblurred image
    deconvolved = np.real(ifftshift(ifft2(deconvolved_fft)))

    return deconvolved


# %% RICHARDSON LUCY DECONVOLUTION

def richardsonlucy(blurred, psf, iterations=20):
    deconvolved = blurred

    for i in range(iterations):
        forwardconv = convolve(deconvolved, psf, 'same')
        ratio = blurred / forwardconv
        deconvolved = deconvolved * convolve(ratio, psf[::-1, ::-1], 'same')

    return deconvolved
 

# %% Choose a deconvolution method and run 

deconvolve_function = richardsonlucy

deconvolved = deconvolve_function(image, psf)

plt.figure()
plt.imshow(deconvolved)
plt.colorbar()
plt.title('Image deconvolved with ' + deconvolve_function.__name__)
plt.show()