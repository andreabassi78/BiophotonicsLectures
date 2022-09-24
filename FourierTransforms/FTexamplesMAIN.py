'''
Created on 2 ago 2019
Shows examples of calculation of 2D Fourier Transform
@author: Andrea Bassi, Politecnico di Milano
Lecture on 2D Fourier Transforms and filtering
Optical Microscopy Course (Biophotonics)
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.fft import fftshift, ifftshift, fft2
from Function2D import function2D

xymin = -5.0 #
xymax = +5.0 #
Nsamples = 200

deltaxy = (xymax-xymin)/Nsamples
x = y = np.linspace(xymin, xymax, Nsamples)
X, Y = np.meshgrid(x,y)

kxymin = -1/(2*deltaxy)
kxymax = 1/(2*deltaxy)
deltakxy = 1/(xymax-xymin)
kx = ky = np.arange(kxymin, kxymax, deltakxy) # not used

# create 2D function

f = function2D(X,Y)
X0 = 0 #translation in x
Y0 = 0 #translation in y
f.setcenter(X0, Y0)
ftype = 'square' #change ftype to choose between 'circle', 'annulus','square','gaussian','sine',deltas
f.functiontype(ftype,0.051)
Z = f.data #Z are the values of the function


# calculate 2D fft (translated to the center using fftshift. ifftshift is used to remove phase errors)
ft=fftshift(fft2(ifftshift(Z)))

# %% create figures
#plot 2D function
fig1 = plt.figure(figsize=(9, 9))

plt.title(f.title)
#plt.title("2D function (Real part)") 

plt.imshow(np.real(Z), interpolation='none', cmap=cm.RdYlGn,
               origin='lower', extent=[xymin,xymax, xymin,xymax],
               vmax=np.real(Z).max(), vmin=-np.real(Z).max())
plt.colorbar()


# plot fft figure
fig2 = plt.figure(figsize=(9, 9))

plt.title(f.title)
plt.axis('off')

ax=[]
# ax enables access to manipulate each of subplots

# create subplot 1: Magnitude of the 2D fft
ax.append(fig2.add_subplot(1, 2, 1))
ax[0].set_title("2D Fourier Transform (Magnitude)")  # set title
plt.imshow(np.abs(ft), interpolation='none', cmap=cm.RdYlGn,
               origin='lower', extent=[kxymin,kxymax, kxymin,kxymax],
               vmax=abs(ft).max(), vmin=-abs(ft).max())

# create subplot 2: Real part of the 2D fft
ax.append(fig2.add_subplot(1, 2, 2))
ax[1].set_title("2D Fourier Transform (Real Part)")  # set title
plt.imshow((np.real(ft)), interpolation='none' , cmap=cm.RdYlGn,
               origin='lower', extent=[kxymin,kxymax, kxymin,kxymax],
               vmax=abs(ft).max(), vmin=-abs(ft).max())

# finally, render the plots

plt.show()

