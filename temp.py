import numpy as np

from numpy.fft import fft2, ifft2, ifftshift, fftshift

import matplotlib.pyplot as plt

 
def free_space_propagator(x,y,z,wavelength,n):
    k = n / wavelength
    r = np.sqrt(x**2+y**2+z**2)
    return - 1.j*k*z* (1+1.j/(2*pi*k*r))*np.exp(1.j*2*pi*k*r)/r**2




 

def convolution(a,b):

    A=fftshift(fft2(ifftshift(a)))

    B=fftshift(fft2(ifftshift(b)))

    C=A*B

    c=ifft2(fftshift(C))

    c=ifftshift(c)

    return c

 

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

# (Abbe's condition is valid because microscope objectives are designed to be

#aberration's free )

 

# define the space at the pupil

b = 15 * mm

xP = yP = np.linspace(-b, +b, Npixels)

XP, YP = np.meshgrid(xP,yP)


kx = XP * k / f

ky = YP * k / f # from x,y to k-space


k_perpendicular = np.sqrt(kx**2 + ky**2) # k perpendicular

k_cut_off = NA/wavelength # cut off frequency in the coherent case


# create a constant ATF

ATF = np.ones([Npixels, Npixels])                  


# cut frequencies outside of the cut off

cut_idx = (k_perpendicular >= k_cut_off)

ATF[cut_idx] = 0


 

ASF=ifftshift(ifft2(fftshift(ATF)))*k**2/f**2   #pay attention to use direct or inverse FT

PSF=np.abs(ASF)**2/k**2

 

dx=dy=(1/np.max(kx))*0.5  #maximum frequency gives me minimun dx (pixel) representable

#this is like when we have defined resolution according to Abbe

 

xmin=-Npixels*dx/2

xmax=Npixels*dx/2

ymin=-Npixels*dy/2

ymax=Npixels*dy/2

 

 
z=6*um

xf = yf = np.linspace(xmin, xmax, Npixels)

XF, YF = np.meshgrid(xf,yf)

D = free_space_propagator(XF,YF,z,wavelength,n)

ASF_out=convolution(ASF,D)

PSF_out=np.abs(ASF_out)**2

print(-np.amin(D)/np.amax(D))
 
plt.figure()

plt.imshow(np.abs(PSF_out),extent = [xmin,xmax,ymin,ymax])

plt.xlabel('x (um)')

plt.ylabel('y (um)')

plt.title('PSF out')


plt.show()