import numpy as np
import matplotlib.pyplot as plt

from numpy import pi
from numpy.fft import fft2,ifft2,fftshift,ifftshift

def D(x,y,z,wavelength,n):
    k = n / wavelength
    r = np.sqrt(x**2+y**2+z**2)
    return - 1.j*k*z* (1+1.j/(2*pi*k*r))*np.exp(1.j*2*pi*k*r)/r**2
    
um = 1.0 

L = 300*um
z = 100*um
wavelength = 0.532*um
n = 1

x = np.linspace(-L/2,L/2, 1024)
y = np.linspace(-L/2,L/2, 1024)

X,Y = np.meshgrid(x,y)

# let's define a plane wave propagating along z
#E0 = np.ones([1024,1024])

# let's define an electric field, spatially modulated with a frequency ky 
kx = 0
ky = 0.1 # 1/um
# E0 = np.exp(-1.j*2*pi*(ky*Y+kx*X))
E0 = np.cos(2*pi*(ky*Y+kx*X))

# let's put a mask with a certain side
side = 30*um
indexes = (np.abs(X)>side/2) | (np.abs(Y)>side/2)
E0[indexes]= 0

plt.figure()
plt.imshow(np.abs(E0))
plt.show()

Dz = D(X,Y,z,wavelength,n)

ft_E0 = fftshift(fft2(E0))
ft_Dz = fftshift(fft2(Dz))

ft_E = ft_E0*ft_Dz

E = ifftshift(ifft2(ft_E))

plt.figure()
plt.imshow(np.abs(E))
plt.show()



















