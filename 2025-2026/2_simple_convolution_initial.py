import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import os

folder = os.path.dirname(os.path.realpath(__file__))
filename = 'nyc.jpg'

f = mpimg.imread(os.path.join(folder,filename))

F = fftshift(fft2(f))

x = y = np.linspace(-1.0, 1.0, 532)
X,Y = np.meshgrid(x,y)

w = 0.02
#define a Gaussian function
g = np.exp((-X**2-Y**2)/w**2)


G = fftshift(fft2(ifftshift(g))) # Transfer function

#G_magnitute = np.abs(G)

#G_inverted = 1 - G_magnitute/np.amax(G_magnitute)  # create a high pass filter

C = G*F

c = ifft2(C)


plt.figure()
plt.imshow(np.abs(f))
plt.title('Original image')
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(np.abs(c))
plt.title('Filtered image')
plt.colorbar()
plt.show()
