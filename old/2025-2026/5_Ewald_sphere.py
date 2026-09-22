import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from matplotlib import pyplot as plt

N = 256  # the number of voxel in N**3

um = 1.0   # base unit is um

n = 1.0     # refractive index
NA = 0.8 # numerical aperture
wavelength = 0.532 * um
k = n / wavelength

kx = ky = kz = np.linspace(-3*k,+3*k, N) # create a k-space 3 times larger than the Ewald sphere radius
KX, KY, KZ = np.meshgrid(kx,ky,kz)     
dK = kx[1]-kx[0] # sampling, in K space


#create the Ewald sphere
ATF = np.zeros((N,N,N))     
ext_radius = k + dK/2 # numerically the Ewald sphere cannot be infinitely thin, so we give it a thickness of dK
int_radius = k - dK/2
indexes = (np.sqrt((KX)**2+(KY)**2+(KZ)**2)<ext_radius) & (np.sqrt((KX)**2+(KY)**2+(KZ)**2)>int_radius) 
ATF[indexes] = 1

# set the NA limit. alternatively one can project a pupilar function on the ATF
k_perpendicular = np.sqrt(KX**2 + KY**2)
indexes = (KZ<0) | (k_perpendicular > NA / wavelength)
ATF[indexes] = 0

# Calculate the Spread and Transfer Functions

ASF = ifftshift(ifftn(fftshift(ATF))) # 3D Amplitude Spread Function 

PSF = np.abs(ASF)**2 # 3D Point Spread Function

OTF = fftshift(fftn(ifftshift(PSF))) # 3D Optical Transfer Function

plane=round(N/2)

epsilon = 1e-9 # to avoid calculating log 0 later

# ATF OTF are shown in the kx-kz plane at ky=0, while ASF and PSF are shown in the x-z plane at y=0
ATF_show = np.rot90( ( np.abs(ATF[plane,:,:]) ) ) #rotation is only to show the z axis vertical
ASF_show = np.rot90( ( np.abs(ASF[plane,:,:]) ) )
PSF_show = np.rot90( ( np.abs(PSF[plane,:,:]) ) )
OTF_show = np.rot90( 10*np.log10 ( np.abs(OTF[plane,:,:]) + epsilon ) ) 

fig, axs = plt.subplots(2,2,figsize=(9,9))

x = y = z = np.fft.fftfreq(N, dK) # finds the spatail coordinates corresponding to the given k-space

kmin = np.amin(kx)
kmax = np.amax(kx)
xmin = np.amin(x)
xmax = np.amax(x)  

# create subplot:
axs[0,0].set_title("|ASF(x,0,z)|")  
axs[0,0].set(ylabel = 'z ($\mu$m)')
axs[0,0].imshow(ASF_show, extent=[xmin,xmax,xmin,xmax])

# # create subplot:
axs[0,1].set_title("|ATF($k_x$,0,$k_z$)|") 
axs[0,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[0,1].imshow(ATF_show, extent=[kmin,kmax,kmin,kmax])

# # create subplot:
axs[1,0].set_title('|PSF(x,0,z)|')  
axs[1,0].set(xlabel = 'x ($\mu$m)')
axs[1,0].set(ylabel = 'z ($\mu$m)')
axs[1,0].imshow(PSF_show, extent=[xmin,xmax,xmin,xmax])

# # create subplot:
axs[1,1].set_title('log|OTF($k_x$,0,$k_z$)|')  
axs[1,1].set(xlabel = '$k_x$ (1/$\mu$m)')
axs[1,1].set(ylabel = '$k_z$ (1/$\mu$m)')
axs[1,1].imshow(OTF_show, extent=[kmin,kmax,kmin,kmax])

plt.show()