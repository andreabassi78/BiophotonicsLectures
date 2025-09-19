import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import os

def convolve(X,Y):
    FX = fft2(X)
    FY = fft2(Y)
    return ifftshift(ifft2(FX*FY))

# open an image that will be used as the phase of the object. 
# Note that the image "filename" must be in the same folder of this .py file
full_path = os.path.realpath(__file__)
folder, _ = os.path.split(full_path) 
filename = 'fibers.tif'
phase = plt.imread(os.path.join(folder,filename))

# normalize the phase and set the wavefront error to a max value of lambda/10
phase = phase/np.amax(phase) * np.pi/10  

# create electric field at the object
E0 = np.exp(1.j*phase)

# create an Amplitude Transfer Function (rescaled version of the pupil) and generate the 2D PSF
um = 1
NA = 0.3
wavelength = 0.500 * um
n = 1 
k = n/wavelength 
Nx, Ny = phase.shape # number of pixels of the image
ATF = np.zeros([Nx, Ny], dtype=np.complex128)
kx = np.linspace(-k, k, Nx)
ky = np.linspace(-k, k, Ny)
KX, KY = np.meshgrid(kx, ky)
dx = 1/2/k
dy = 1/2/k
x = np.linspace (- dx*Nx/2, + dx*Nx/2, Nx)
y = np.linspace (- dy*Ny/2, + dy*Ny/2, Ny)
k_cut_off = NA/wavelength
indexes = np.sqrt(KX**2+KY**2)<k_cut_off
ATF[indexes] = 1

# add a Zernike pupil
epsilon = 0.05*k_cut_off
indexes = np.sqrt(KX**2+KY**2) < epsilon
transmittance = 0.6
ATF[indexes] = 1.j *transmittance
#

# add a Schlieren pupil
indexes = KX <0
# ATF[indexes] = 0
#

# Calculate the Amplitude Spread Function
ASF = ifftshift(ifft2(fftshift(ATF)))

# Calculate the image at the detector
E1 = convolve(E0, ASF)
image = np.abs(E1)**2

# show images
fig, axs = plt.subplots(2, 2)

axs[0,0].set_title("Amplitude Transfer Function (Pupil), real part")
im0 = axs[0,0].imshow(np.real(ATF), cmap='gray',extent = [np.amin(kx),np.amax(kx),np.amin(ky),np.amax(ky)])
axs[0,0].set_ylabel('ky (1/um)')
plt.colorbar(im0, ax=axs[0, 0])

axs[0,1].set_title("Amplitude Spread Function, real part")
im1 = axs[0,1].imshow(np.real(ASF), cmap='gray', extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)])
axs[0,1].set_ylabel('y (um)')
plt.colorbar(im1, ax=axs[0, 1])

axs[1,0].set_title("Original phase of the object (rad)")
im2 = axs[1,0].imshow(phase, cmap='gray',extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)]) 
axs[1,0].set_xlabel('x (um)')
axs[1,0].set_ylabel('y (um)')
plt.colorbar(im2, ax=axs[1, 0])

axs[1,1].set_title("Image intensity at the detector")
im3 = axs[1,1].imshow(np.abs(image), cmap='gray', extent = [np.amin(x),np.amax(x),np.amin(y),np.amax(y)])
axs[1,1].set_ylabel('y (um)')
axs[1,1].set_xlabel('x (um)')
plt.colorbar(im3, ax=axs[1, 1])

plt.show()
