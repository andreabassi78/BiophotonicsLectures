'''
Created on 28 jul 2019
@author: Andrea Bassi, Politecnico di Milano
Lecture on 3D Optical Transfer Functions and Ewald Sphere. 
Optical Microscopy Course (Biophotonics)
'''

import numpy as np 
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
from AmplitudeTransferFunction_3D import amplitude_transfer_function

N = 200  # the number of voxel in N**3
um = 1.0   # base unit is um
n = 1.0     # refractive index
NA = 0.8 # numerical aperture
wavelength = 0.520 * um
K = n / wavelength

Detection_Mode = 'stardard'
# choose between 'standard' and '4pi'

Microscope_Type = 'aberrated'
# choose between: 'widefield', 'gaussian', 'bessel', 'SIM', 'STED', 'aberrated' 

# generate the Amplitude Transfer Function (also called Coherent Transfer Function)
Kextent = 3*K  # extent of the k-space
H = amplitude_transfer_function(N, Kextent, n)
H.create_ewald_sphere(K)
H.set_numerical_aperture(NA, Detection_Mode)
H.set_microscope_type(NA, Microscope_Type)

# calculate the Spread and Transfer Function
ATF = H.values # 3D Amplitude Transfer Function

ASF = ifftshift(ifftn(fftshift(ATF))) # 3D Amplitude Spread Function 

PSF = np.abs(ASF)**2 # 3D Point Spread Function
PSF = PSF / np.sum(PSF) # Normalize the PSF on its eneregy

OTF = fftshift(fftn(ifftshift(PSF))) # 3D Optical Transfer Function

# show figures
plane=round(N/2)
epsilon = 1e-9 # to avoid calculating log 0 later
ATF_show = np.rot90( ( np.abs(ATF[plane,:,:]) ) )
ASF_show = np.rot90( ( np.abs(ASF[plane,:,:]) ) )
PSF_show = np.rot90( ( np.abs(PSF[plane,:,:]) ) )
OTF_show = np.rot90( 10*np.log10 ( np.abs(OTF[plane,:,:]) + epsilon ) ) 

# set font size
plt.rcParams['font.size'] = 12

#create figure 1 and subfigures
fig1, axs = plt.subplots(2,2,figsize=(9,9))
fig1.suptitle(Detection_Mode + ' ' + Microscope_Type + ' microscope')

# recover the extent of the axes x,y,z
rmin = H.rmin
rmax = H.rmax

# create subplot:
axs[0,0].set_title("|ASF(x,0,z)|")  
axs[0,0].set(ylabel = 'z ($\mu$m)')
axs[0,0].imshow(ASF_show, extent=[rmin,rmax,rmin,rmax])

# create subplot:
axs[0,1].set_title("|ATF($k_x$,0,$k_z$)|") 
axs[0,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[0,1].imshow(ATF_show, extent=[-Kextent,Kextent,-Kextent,Kextent])

# create subplot:
axs[1,0].set_title('|PSF(x,0,z)|')  
axs[1,0].set(xlabel = 'x ($\mu$m)')
axs[1,0].set(ylabel = 'z ($\mu$m)')
axs[1,0].imshow(PSF_show, extent=[rmin,rmax,rmin,rmax])

# create subplot:
axs[1,1].set_title('log|OTF($k_x$,0,$k_z$)|')  
axs[1,1].set(xlabel = '$k_x$ (1/$\mu$m)')
axs[1,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[1,1].imshow(OTF_show, extent=[-Kextent,Kextent,-Kextent,Kextent])


# zoom in:
# for i in (0,1): 
#     for j in (0,1):
#         if j == 1:
#             zoom_factor = Kextent/K/2
#         else:
#             zoom_factor = Kextent/K
#         axs[i,j].xaxis.zoom(zoom_factor) 
#         axs[i,j].yaxis.zoom(zoom_factor)
        
    
# finally, render the figures
plt.show()    

print('The numerical aperture of the system is:', NA) 
print('The transverse resolution is:', wavelength/2/NA ,'um') 
if Detection_Mode == 'standard':
    print('The axial resolution is:', wavelength/n/(1-np.sqrt(1-NA**2/n**2)) ,'um') 
    print('The axial resolution is:', 2*n*wavelength/NA**2 ,'um, with Fresnel approximation') 

