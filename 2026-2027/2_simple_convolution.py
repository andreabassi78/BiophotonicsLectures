import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import os

folder = os.path.dirname(os.path.realpath(__file__))
filename = 'nyc.jpg'

f = mpimg.imread(os.path.join(folder,filename))

F = fftshift(fft2(f))

# define a Gaussian kernel for convolution
x = y = np.linspace(-1, 1, 532)
X, Y = np.meshgrid(x, y)
w = 0.02
g = np.exp(-((X**2 + Y**2)/(w**2)))

# make the kernel negative on the left half: use this to detect edges in the image
# g[:,0:266] = -g[:,0:266] 

G = fftshift(fft2(g))

R = F * G

r = ifftshift(ifft2(R)) #filtered function in the spatial domain

plt.figure()
plt.imshow(np.abs(f))
plt.title('Original image')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.log(np.abs(F)+1e-10))
plt.title('Fourier transform of the image')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(g)
plt.title('Kernel for convolution')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.log(np.abs(G)+1e-10))
plt.title('Fourier transform of the kernel (transfer function)')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.abs(r))
plt.title('Filtered image')
plt.colorbar()
plt.show()