# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 23:00:29 2021
Creates a 3D PSF starting from circular pupil
@author: Andrea Bassi
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt
from skimage import io, color, transform
import matplotlib.pyplot as plt

# Path to your image (can be local or URL)
img_path = "Add the path of your image here"

um = 1.0
mm = 1000.0
pi = np.pi

#Check Npixels with value of 1 and 10
Npixels = 128*20 # Pixels in x,y and number of planes z
n = 1 # refractive index
wavelength = 0.532*um 
f = 10*mm # focal length of the objective lens
a = 4*mm  # radius of the the pupil
k = n/wavelength # wavenumber
NA = n*(a/f) # Numerical aperture

#-----------------------------------------------
def load_and_prepare_image(img_path, ATF):
    # 1. Load image
    img = io.imread(img_path)

    # 2. Convert to grayscale safely
    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA → RGB
            img = img[..., :3]
        img = color.rgb2gray(img)
    elif img.ndim == 2:
        pass  # already grayscale
    else:
        raise ValueError("Unsupported image format!")

    # 3. Resize image to match ATF size
    target_shape = ATF.shape
    img_resized = transform.resize(img, target_shape, anti_aliasing=True)

    # 4. Normalize intensity
    U0 = img_resized / np.max(img_resized)

    return U0
#---------------------------------------
# define the space at the pupil
b = 15 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

# calculate spatial frequencies at the pupil plane
kx = XP * k / f # assuming Abbe's sine condition
ky = YP * k / f # assuming Abbe's sine condition

k_perpendicular = np.sqrt(kx**2 + ky**2) # k perpendicular
k_cut_off = NA/wavelength # cut off frequency in the coherent case

#--------------------------------------  Different ATF options to choose ----------------------------------
# create a constant ATF
ATF = np.ones([Npixels, Npixels])  
# ATF for astigmatism
#ATF = np.ones([Npixels, Npixels])                  
rho   = k_perpendicular / k_cut_off
theta = np.arctan2(ky, kx)
# Zernike components
Z22  = rho**2 * np.cos(2 * theta)
Z2m2 = rho**2 * np.sin(2 * theta)
# Choose amplitude in waves
A_cos = 0.8   # Z_2^2 strength (x/y astigmatism)
A_sin = 0   # Z_2^-2 strength (45° astigmatism)
# Combine them (general astigmatism)
W_ast = A_cos * Z22 + A_sin * Z2m2
# Phase in radians
phi_ast = 2 * np.pi * W_ast
# Apply to ATF
ATF = ATF * np.exp(1j * phi_ast)
# add defocus
z = -5.0*um
kz = np.sqrt(k**2-k_perpendicular**2)
angular_spectrum_propagator = np.exp(1.j*2*pi*kz*z)
# cut frequencies outside of the cut off (I can multiply by 2 or divide by 10)
cut_idx = (k_perpendicular >= k_cut_off*2) 
ATF = ATF * angular_spectrum_propagator
ATF[cut_idx] = 0

'''
#Circular pupil low-pass filter with defocus
# create a constant ATF
ATF = np.ones([Npixels, Npixels])    
# add defocus
z = -5.0*um
kz = np.sqrt(k**2-k_perpendicular**2)
angular_spectrum_propagator = np.exp(1.j*2*pi*kz*z)
# cut frequencies outside of the cut off (I can multiply by 2 or divide by 10)
cut_idx = (k_perpendicular >= k_cut_off*2) 
ATF = ATF * angular_spectrum_propagator
ATF[cut_idx] = 0


#Circular pupil High-pass filter, divide cutoff by 20 to 100
# create a constant ATF
ATF = np.zeros([Npixels, Npixels])    
cut_idx = (k_perpendicular >= k_cut_off/100) 
ATF[cut_idx] = 1
x

#Annular pupil to try with cutoff /100 and by /50
# create a constant ATF
ATF = np.zeros([Npixels, Npixels])    
# cut frequencies outside of the cut off (I can multiply by 2 or divide by 10)
ATF[(k_perpendicular <= k_cut_off/100) & (k_perpendicular >= 0.6*k_cut_off/100)] = 1
'''


#-------------------------------------------------------------------------

# Reconstruct ATF before cutoff and plot before/after side by side
ATF_before = np.ones_like(ATF) * angular_spectrum_propagator

extent_k = [np.amin(kx), np.amax(kx), np.amin(ky), np.amax(ky)]
fig_atf_compare, axs = plt.subplots(2, 2, figsize=(10, 8))
ax_bmag, ax_bphase, ax_amag, ax_aphase = axs.ravel()

im_bmag = ax_bmag.imshow(np.abs(ATF_before), extent=extent_k, origin='lower')
ax_bmag.set_title('ATF magnitude (before cutoff)')
ax_bmag.set_xlabel('kx (1/um)'); ax_bmag.set_ylabel('ky (1/um)')
plt.colorbar(im_bmag, ax=ax_bmag, fraction=0.046, pad=0.04)

im_bphase = ax_bphase.imshow(np.angle(ATF_before), extent=extent_k, origin='lower', cmap='twilight')
ax_bphase.set_title('ATF phase (before cutoff)')
ax_bphase.set_xlabel('kx (1/um)')
plt.colorbar(im_bphase, ax=ax_bphase, fraction=0.046, pad=0.04)

im_amag = ax_amag.imshow(np.abs(ATF), extent=extent_k, origin='lower')
ax_amag.set_title('ATF magnitude (after cutoff)')
ax_amag.set_xlabel('kx (1/um)'); ax_amag.set_ylabel('ky (1/um)')
plt.colorbar(im_amag, ax=ax_amag, fraction=0.046, pad=0.04)

im_aphase = ax_aphase.imshow(np.angle(ATF), extent=extent_k, origin='lower', cmap='twilight')
ax_aphase.set_title('ATF phase (after cutoff)')
ax_aphase.set_xlabel('kx (1/um)')
plt.colorbar(im_aphase, ax=ax_aphase, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------
# Image usage

# Load and prepare the image
U0 = load_and_prepare_image(img_path, ATF)

plt.imshow(U0, cmap='gray')
plt.title("Prepared image (normalized, resized to ATF shape)")
plt.axis('off')
plt.show()


# 1. Fourier transform the input image
FU0 = fftshift(fft2(ifftshift(U0)))
FU1 = ATF * FU0
#----------------------------------------------------------------------

U1 = ifftshift(ifft2(fftshift(FU1))) # Amplitude Spread Function   
I1 = np.abs(U1)**2 # Point Spread Function  

# calculate the space at the object plane
dr = 1/2/np.amax(kx)
x = y = np.linspace (- dr*Npixels/2, + dr*Npixels/2, Npixels)


fig0, ax0 = plt.subplots()
ax0.imshow(I1, cmap='gray', extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)])
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title(f'|PSF(x,y,z={z}um)|')

plt.show()


