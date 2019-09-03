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
from EwaldSphere.AmplitudeTransferFunction import amplitude_transfer_function
from xlrd.formula import num2strg

Kmax = 4.0    # maximum value of K in the K space
N = 300     # sampling number
K = 1.0     # radius of the Ewald sphere K=n/lambda

NA = 0.7   # numerical aperture
n = 1.0       # refractive index

Detection_Mode = 'standard'
#choose between 'standard' and '4pi'
Microscope_Type = 'widefield'
# choose between: 'widefield', 'gaussian', 'bessel', 'SIM', 'STED', 'aberrated' 

# Generate the Amplitude Transfer Function (or Coherent Transfer Function)
t0=time.time() # this is to calculate the execution time

H = amplitude_transfer_function(N, Kmax, n)
H.create_ewald_sphere(K)
H.set_numerical_aperture(NA, Detection_Mode)
pupil, psf_xy0 = H.set_microscope_type(NA, Microscope_Type)


ATF = H.values # Amplitude Transfer Function

ASF = ifftshift(ifftn(fftshift(ATF))) * N**3 #
# Amplitude Spread Function (normalized for the volume of the space)

PSF = np.abs(ASF)**2 # Point Spread Function

OTF = fftshift(fftn(ifftshift(PSF))) # Optical Transfer Function

print('Elapsed time for calculation: ' + num2strg( time.time()-t0) )

plane=round(N/2)
epsilon = 1e-9 # to avoid calculating log 0 later
ATF_show = np.rot90( ( np.abs(ATF[plane,:,:]) ) )
ASF_show = np.rot90( ( np.abs(ASF[plane,:,:]) ) )
PSF_show = np.rot90( ( np.abs(PSF[plane,:,:]) ) )
OTF_show = np.rot90( 10*np.log10 ( np.abs(OTF[plane,:,:]) + epsilon ) ) 

############################################################################################
#####    Create figures

# set font size
plt.rcParams['font.size'] = 12

#create figure and subfigures
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
axs1[0,1].imshow(ATF_show ,  extent=[-Kmax*n,Kmax*n,-Kmax*n,Kmax*n])

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
axs1[1,1].imshow(OTF_show, extent=[-Kmax*n,Kmax*n,-Kmax*n,Kmax*n])

#fig1.tight_layout() # prevents overlap of y-axis labels

fig2, axs2 = plt.subplots( 1, 2, figsize=(8,8) )
fig2.suptitle(Detection_Mode + ' ' + Microscope_Type + ' microscope')

axs2[0].set_title('|ASF(x,y,0)|')  
axs2[0].set(xlabel = '$x/\lambda$')
axs2[0].set(ylabel = '$y/\lambda$')
axs2[0].imshow(np.abs(np.rot90(np.abs(psf_xy0)) ), extent=[-Rmax/n,Rmax/n,-Rmax/n,Rmax/n])

axs2[1].set(xlabel = '$k_x\lambda$')
axs2[1].set(ylabel = '$k_y\lambda$')
if Microscope_Type in ('STED' , 'aberrated'): 
    ims=axs2[1].imshow(np.rot90(np.angle(pupil)),
                       extent=[-Kmax*n,Kmax*n,-Kmax*n,Kmax*n]) #plot the amplitude of the pupil
    axs2[1].set_title('\u2220 Pupil')  
else:
    ims=axs2[1].imshow(np.rot90(np.abs(pupil)),
                       extent=[-Kmax*n,Kmax*n,-Kmax*n,Kmax*n]) #plot the phase of the pupil
    axs2[1].set_title('|Pupil|')  
    
# fig2.tight_layout() # prevents overlap of y-axis labels
fig2.colorbar(ims, ax=axs2[1])
# finally, render the figures
plt.show()