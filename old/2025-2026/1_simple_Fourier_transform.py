"""
Example of a simple 2D Fourier Transform
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

N = 101
x = y = np.linspace(-1.0, 1.0, N)
X,Y = np.meshgrid(x,y)

#w = 0.4
#define a Gaussian function
#z = np.exp((-X**2-Y**2)/w**2)

# define a rectangle function

z = np.zeros([N,N])
side = 0.3
indexes = (np.abs(X)<side/2) & (np.abs(Y)<side/2)
z[indexes] = 1

plt.figure()
plt.imshow(z)
plt.show()

Fz = fftshift(fft2(z))

plt.figure()
plt.imshow(np.abs(Fz))
plt.show()

