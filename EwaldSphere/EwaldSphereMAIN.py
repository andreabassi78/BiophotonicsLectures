'''
Created on 28 jul 2019
@author: Andrea Bassi, Politecnico di Milano
Lecture on 3D Optical Transfer Functions and Ewald Sphere. 
Optical Microscopy Course (Biophotonics)
'''

import numpy as np 
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
import time
from AmplitudeTransferFunction import amplitude_transfer_function
from xlrd.formula import num2strg

Kextent = 2  # maximum value of Kx,Ky,Kz in the K space. k space goes from -Kextent to +Kextent
N = 256        # sampling number
K = 1        # radius of the Ewald sphere K=n/lambda
NA = 0.3
       # numerical aperture
n = 1.        # refractive index

Detection_Mode = 'standard'
#choose between 'standard' and '4pi'

Microscope_Type = 'widefield'
# choose between: 'widefield', 'gaussian', 'bessel', 'SIM', 'STED', 'aberrated' 

SaveData = False

############################
############################

# Generate the Amplitude Transfer Function (or Coherent Transfer Function)
t0=time.time() # this is to calculate the execution time

H = amplitude_transfer_function(N, Kextent, n)
pixel_size = H.dr
extent = H.xyz_extent

print ('Real space sampling = ' + num2strg(pixel_size/n) + ' * wavelength')

H.create_ewald_sphere(K)
H.set_numerical_aperture(NA, Detection_Mode)
pupil, psf_xy0 = H.set_microscope_type(NA, Microscope_Type)

ATF = H.values # 3D Amplitude Transfer Function

ASF = ifftshift(ifftn(fftshift(ATF))) * N**3 #
# 3D Amplitude Spread Function (normalized for the total volume)

PSF = np.abs(ASF)**2 # 3D Point Spread Function

OTF = fftshift(fftn(ifftshift(PSF))) # 3D Optical Transfer Function

print('Elapsed time for calculation: ' + num2strg( time.time()-t0) + 's' )

plane=round(N/2)
epsilon = 1e-9 # to avoid calculating log 0 later
ATF_show = np.rot90( ( np.abs(ATF[plane,:,:]) ) )
ASF_show = np.rot90( ( np.abs(ASF[plane,:,:]) ) )
PSF_show = np.rot90( ( np.abs(PSF[plane,:,:]) ) )
OTF_show = np.rot90( 10*np.log10 ( np.abs(OTF[plane,:,:]) + epsilon ) ) 

############################
#####    Create figures

def colorbar(mappable):
    '''Auxiliary function to plot the colorbar in scale'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

# set font size
plt.rcParams['font.size'] = 12

#create figure 1 and subfigures
fig1, axs1 = plt.subplots( 2, 2, figsize= (8,8) )
fig1.suptitle(Detection_Mode + ' ' + Microscope_Type + ' microscope')

# Recover extent of axes x,y,z (it is the inverse of the k sampling xyz_extent = 1/(2*dK))
Rmax=H.xyz_extent

#data are show as multiples of lambda, therefore multiplication or division by n is required below

# create subplot:
axs1[0,0].set_title("|ASF(x,0,z)|")  
#axs[0,0].set(xlabel = '$x/\lambda$')
axs1[0,0].set(ylabel = '$z/\lambda$')
axs1[0,0].imshow(ASF_show, vmin = ASF_show.min(), vmax = ASF_show.max(), 
                 extent=[-Rmax/n,Rmax/n,-Rmax/n,Rmax/n])

# create subplot:
axs1[0,1].set_title("|ATF($k_x$,0,$k_z$)|") 
#axs[0,1].set(xlabel = '$k_x\lambda$')
axs1[0,1].set(ylabel = '$k_z\lambda$')
axs1[0,1].imshow(ATF_show ,  extent=[-Kextent*n,Kextent*n,-Kextent*n,Kextent*n])

# create subplot:
axs1[1,0].set_title('|PSF(x,0,z)|')  
axs1[1,0].set(xlabel = '$x/\lambda$')
axs1[1,0].set(ylabel = '$z/\lambda$')
axs1[1,0].imshow(PSF_show, vmin = PSF_show.min(), vmax = PSF_show.max(), 
                 extent=[-Rmax/n,Rmax/n,-Rmax/n,Rmax/n])

# create subplot:
axs1[1,1].set_title('log|OTF($k_x$,0,$k_z$)|')  
axs1[1,1].set(xlabel = '$k_x\lambda$')
axs1[1,1].set(ylabel = '$k_z\lambda$')
axs1[1,1].imshow(OTF_show, extent=[-Kextent*n,Kextent*n,-Kextent*n,Kextent*n])

#fig1.tight_layout() # prevents overlap of y-axis labels

#create figure 2 and subfigures
fig2, axs2 = plt.subplots( 1, 2, figsize=(8,8) )
fig2.suptitle(Detection_Mode + ' ' + Microscope_Type + ' microscope')

axs2[0].set_title('|ASF(x,y,0)|')  
axs2[0].set(xlabel = '$x/\lambda$')
axs2[0].set(ylabel = '$y/\lambda$')
axs2[0].imshow(np.abs((np.abs(psf_xy0)) ), extent=[-Rmax/n,Rmax/n,-Rmax/n,Rmax/n])

axs2[1].set(xlabel = '$k_x\lambda$')
axs2[1].set(ylabel = '$k_y\lambda$')
if Microscope_Type in ('STED' , 'aberrated'): 
    ims=axs2[1].imshow((np.angle(pupil)),
                       extent=[-Kextent*n,Kextent*n,-Kextent*n,Kextent*n]) #plot the amplitude of the pupil
    axs2[1].set_title('\u2220 Pupil') #angle symbol \u2220 
else:
    ims=axs2[1].imshow((np.abs(pupil)),
                       extent=[-Kextent*n,Kextent*n,-Kextent*n,Kextent*n]) #plot the phase of the pupil
    axs2[1].set_title('|Pupil|')  
    
# fig2.tight_layout() # prevents overlap of y-axis labels
colorbar(ims)

# finally, render the figures
plt.show()

############################
##### Save Psf to .tif file

if SaveData:
    
    wavelength = 0.520 #um  #you may want to specify a wavelength to save a calibrated PSF
    
    from skimage.external import tifffile as tif
    psf16 = np.transpose(np.abs(PSF),(2,0,1))
    psf16 = ( psf16 * (2**16-1) / np.amax(psf16) ).astype('uint16') #normalize and convert to 16 bit
    psf16.shape = 1, N, 1, N, N, 1 # dimensions in TZCYXS order
    sampling = pixel_size/n*wavelength
    tif.imsave('psf.tif', psf16, imagej=True, resolution = (1.0/sampling, 1.0/sampling),
                metadata={'spacing': sampling, 'unit': 'um'})