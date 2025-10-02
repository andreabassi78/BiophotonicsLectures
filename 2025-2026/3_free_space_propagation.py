import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, cos
from numpy.fft import fft2, ifft2, ifftshift

def kernelRS(x,y,z,wavelength,n):
    k = n/wavelength
    r = sqrt(x**2+y**2+z**2)
    return -1.j*k*z *exp(1.j *2* pi*k*r)/r**2 *(1 + 1.j/(2*pi*k*r) )

um = 1.0
wavelength = 0.532 * um 
n = 1

Nsamples = 1024 # number of pixels
L = 300 * um # extent of the xy space
x = y = np.linspace(-L/2, +L/2, Nsamples)
X, Y = np.meshgrid(x,y)

""" create a constant field E0 (plane wave propagating along z)"""
# E0 = np.ones([Nsamples, Nsamples])


"""create a co-sinusoidal field E0 (interference of 2 plane waves at a certain angle)"""
#kx = 0.2 /um
#ky = 0.0 /um
#E0 = cos(2*pi*(kx*X+ky*Y))

"""create a costant field E0 at a certain angle"""
kx = 0.2 /um
ky = 0.0 /um
E0 = exp(1.j*2*pi*(kx*X+ky*Y))


""" insert a square mask """
side = 30 * um
indexes = (np.abs(X)>side/2) | (np.abs(Y)>side/2)
E0[indexes] = 0

z = 1000*um
D = kernelRS(X,Y,z,wavelength,n)

E1 = ifftshift(ifft2(fft2(E0)*fft2(D)))

plt.imshow(np.abs(E1))
plt.show()