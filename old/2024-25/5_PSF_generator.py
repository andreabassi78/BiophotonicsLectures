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


# add phase due to out of focus propagation 
kz = np.sqrt(k**2-k_perpendicular**2)
z = 6 *um
angular_spectrum_propagator = np.exp(1.j *2 *pi *kz*z)
ATF = ATF *angular_spectrum_propagator


# cut frequencies outside of the cut off
cut_idx = (k_perpendicular >= k_cut_off) 
ATF[cut_idx] = 0

plt.figure() 
plt.imshow(np.abs(ATF),extent = [np.min(kx),np.max(kx),np.min(kx),np.max(kx)])
plt.xlabel('kx (um-1)')
plt.ylabel('ky (um-1)')
plt.title('ATF')

ASF = ifftshift(ifft2(ATF))

PSF = np.abs(ASF)**2

dx = 1/np.max(kx)/2
dy = 1/np.max(ky)/2

xmin = -dx * Npixels/2
xmax = +dx * Npixels/2
ymin = -dy * Npixels/2
ymax = +dy * Npixels/2 

plt.figure() 
plt.imshow(PSF,extent = [xmin,xmax,ymin,ymax])
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title(f'PSF at z = {z} um')

plt.show()