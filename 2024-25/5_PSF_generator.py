import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt

um = 1.0
mm = 1000.0
pi = np.pi

Npixels = 128 # Pixels in x,y and number of planes z
n = 1 # refractive index
wavelength = 0.532*um 
f = 10*mm # focal length of the objective lens
a = 4*mm  # radius of the the pupil
k = n/wavelength # wavenumber
NA = n*(a/f) # Numerical aperture, assuming Abbe's sine condition

# define the space at the pupil
b = 15 * mm 
xP = yP = np.linspace(-b, +b, Npixels)
XP, YP = np.meshgrid(xP,yP)

kx = XP * k / f # assuming Abbe's sine condition
ky = YP * k / f # assuming Abbe's sine condition

k_perpendicular = np.sqrt(kx**2 + ky**2) # k perpendicular
k_cut_off = NA/wavelength # cut off frequency in the coherent case

# create a constant ATF
ATF = np.ones([Npixels, Npixels])                  

# cut frequencies outside of the cut off
cut_idx = (k_perpendicular >= k_cut_off) 
ATF[cut_idx] = 0

plt.figure() 
plt.imshow(ATF,extent = [np.min(kx),np.max(kx),np.min(kx),np.max(kx)])
plt.xlabel('kx (um-1)')
plt.ylabel('ky (um-1)')
plt.title('ATF')

plt.show()