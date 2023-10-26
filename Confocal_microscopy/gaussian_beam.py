

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt

um = 1.0
mm = 1000.0
pi = np.pi

Npixels = 256 # Pixels in x,y and number of planes z
wavelength = 0.500*um 
n = 1
w0 = 0.29 * um

zr = np.pi * w0**2 *n / wavelength
print(zr)
# define the space at the focus plane
a = 8 * um 
b = 8 * um
rho = np.linspace(-a, +a, Npixels)
z = np.linspace(-b, +b, Npixels)
Rho, Z = np.meshgrid(rho,z)

w= w0 * np.sqrt(1+(Z/zr)**2)

I = np.exp(-2*Rho**2/w**2) * (w0/w)**2

PSF = I

plt.gray()
fig0, ax0 = plt.subplots()
ax0.imshow(PSF, extent = [np.amin(rho),np.amax(rho),np.amin(z),np.amax(z)])
plt.xlabel('x (um)')
plt.ylabel('z (um)')
plt.title(f'|PSF(x,z|')

dkx = 1/2/np.amax(rho)
dkz = 1/2/np.amax(z)
kx = np.linspace (- dkx*Npixels/2, + dkx*Npixels/2, Npixels)
kz = np.linspace (- dkz*Npixels/2, + dkz*Npixels/2, Npixels)

OTF = fftshift(fft2(PSF)) # Optical Transfer Function

fig0, ax0 = plt.subplots()
ax0.imshow(np.log10(np.abs(OTF)+10), extent = [np.amin(kx),np.amax(kx),np.amin(kz),np.amax(kz)])
plt.xlabel('kx (1/um)')
plt.ylabel('kz (1/um)')
plt.title(f'log|OTF(kx,kz|')
plt.gray()
plt.show()

