import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import os

folder = os.path.dirname(os.path.realpath(__file__))
filename = 'nyc.jpg'

f = mpimg.imread(os.path.join(folder,filename))

F = fftshift(fft2(f))

# define a kernel for convolution
x = y = np.linspace(-1, 1, 532)
X, Y = np.meshgrid(x, y)
w = 0.05
g = np.exp(-((X**2 + Y**2)/(w**2)))

G = fftshift(fft2(g))


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
plt.imshow(np.abs(g))
plt.title('Kernel for convolution')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.log(np.abs(G)+1e-10))
plt.title('Fourier transform of the kernel (transfer function)')
plt.colorbar()
plt.show()