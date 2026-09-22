"""Calculatest the PSF of a confocal approximating the illumination and detection PSF to gaussian functions
and considering a pinhole at the detection path."""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt

def convolve(X,Y):
    FX = fftshift(fft2(X))
    FY = fftshift(fft2(Y))
    return ifftshift(ifft2(FX*FY))

um = 1.0
mm = 1000.0
pi = np.pi

Npixels = 256 # Pixels in x and z
wavelength = 0.500*um 
n = 1
w0 = 0.400 * um
zr = pi * w0**2 *n / wavelength

# define the space at the focus plane
a = 2 * um 
b = 3 * um
rho = np.linspace(-a, +a, Npixels)
z = np.linspace(-b, +b, Npixels)
Rho, Z = np.meshgrid(rho,z)

w = w0 * np.sqrt(1+(Z/zr)**2)
I = np.exp(-2*Rho**2/w**2) * (w0/w)**2 # Intensity of a Gaussian beam that simulates the illumination and (ideal) detection PSF

pinhole = np.zeros_like(I)
r = 10*um/ 40 # radius of the pinhole at the object. 
# Note that the pinhole size at the image plane is scaled by the magnification factor (40x) because we consider the PSF at the object plane

ph = np.abs(rho) < r
pinhole[Npixels//2, ph] = 1

PSF_illumination = I
PSF_detection = np.abs(convolve(I,pinhole)) 
PSF_conf = PSF_illumination * PSF_detection

fig0, ax0 = plt.subplots()
ax0.imshow(PSF_illumination, extent = [np.amin(rho),np.amax(rho),np.amin(z),np.amax(z)])
plt.xlabel('x (um)')
plt.ylabel('z (um)')
plt.title(f'PSF_illumination(x,z)')

fig0, ax0 = plt.subplots()
ax0.imshow(pinhole, extent = [np.amin(rho),np.amax(rho),np.amin(z),np.amax(z)])
plt.xlabel('x (um)')
plt.ylabel('z (um)')
plt.title(f'pinhole')

fig0, ax0 = plt.subplots()
ax0.imshow(PSF_detection, extent = [np.amin(rho),np.amax(rho),np.amin(z),np.amax(z)])
plt.xlabel('x (um)')
plt.ylabel('z (um)')
plt.title(f'PSF_detection(x,z)')

fig0, ax0 = plt.subplots()
ax0.imshow(PSF_conf, extent = [np.amin(rho),np.amax(rho),np.amin(z),np.amax(z)])
plt.xlabel('x (um)')
plt.ylabel('z (um)')
plt.title(f'PSF_confocal(x,z)')

dkx = 1/2/np.amax(rho)
dkz = 1/2/np.amax(z)
kx = np.linspace (- dkx*Npixels/2, + dkx*Npixels/2, Npixels)
kz = np.linspace (- dkz*Npixels/2, + dkz*Npixels/2, Npixels)

# OTF = fftshift(fft2(PSF_conf)) # Optical Transfer Function
# fig0, ax0 = plt.subplots()
# ax0.imshow(np.abs(OTF), extent = [np.amin(kx),np.amax(kx),np.amin(kz),np.amax(kz)])
# plt.xlabel('kx (1/um)')
# plt.ylabel('kz (1/um)')
# plt.title(f'|OTF(kx,kz|')
# plt.gray()

plt.show()

