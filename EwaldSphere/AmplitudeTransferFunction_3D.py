'''
Created on 28 jul 2019
@author: Andrea Bassi, Politecnico di Milano
'''

import numpy as np
from numpy.fft import ifft2, fftshift, ifftshift
from Function2D import function2D
  
class amplitude_transfer_function(object):
    '''
    Generare an Amplitude Transfer Function (or Coherent Transfer function) from an Ewald sphere,
    projecting a 2D pupil on the sphere
    '''
    def __init__(self, N = 256, K_xyz_extent = 2.0, n = 1.0):
        '''
        Constructor.
        Creates a space KX,KY,KZ. 
        The K space extends between -+ K_xyz_extent
        N**3 is the number of voxels
        n is the refractive index.
         
        Inizializes the ATF (self.values) to zero.
        dK is the sampling step in the K space
        dr is the sampling step in the real space
        
        '''
        self.N = N
        self.n = n
        self.K_xyz_amplitude = K_xyz_extent # half length of the axes Kx,Ky,Kz
        self.values = np.zeros((N,N,N))     
        kx = ky = kz = np.linspace(-K_xyz_extent,K_xyz_extent, N)
        self.KX, self.KY, self.KZ = np.meshgrid(kx,ky,kz)     
        self.dK = kx[1]-kx[0] #sampling, in K space
        x = y = z = np.fft.fftfreq(self.N, self.dK)
        
        self.dr = x[1]-x[0]    # spatial sampling, in real space.
        
        self.rmin = min(x)
        self.rmax = max(x)
        self.microscope_type = None
        
        
    def create_ewald_sphere(self, Kradius = 1.0):
        '''
        Creates a sphere with a certain radius in the K space
        The thickness of the sphere is dK
        '''
        ext_radius = Kradius
        int_radius = Kradius-self.dK
        k = np.sqrt((self.KX)**2+(self.KY)**2+(self.KZ)**2)
        indexes = (k<ext_radius) * (k>int_radius) 
        self.K = Kradius
        self.values[indexes] = 1
    
    def set_numerical_aperture(self, NA, *args):
        '''
        Limits the Ewald Sphere to a certain Numerical Aperture
        '''
        if len(args)>0:
            mode = args[0]
        else:
            mode = self.microscope_type       
        
        Kz_min = 0
        Kxy_max = self.K * NA / self.n
        K_xy = np.sqrt((self.KX)**2+(self.KY)**2)
        
        if mode == '4pi':
            indexes = (K_xy > Kxy_max) 
        else:
            indexes = (self.KZ < Kz_min) | (K_xy > Kxy_max)
        self.values[indexes] = 0
   
    def set_microscope_type(self, NA, mtype):    
        '''
        Projects the pupil on the Ewald Sphere
        '''
        kx = self.KX[:,:,0]
        ky = self.KY[:,:,0]
        pupil = function2D(kx,ky)
        Kxy_max = (self.K) * NA / self.n
        
        if mtype == 'widefield':
            pupil.functiontype('circle', Kxy_max)
        elif mtype == 'gaussian':
            pupil.functiontype('gaussian', Kxy_max)
        elif mtype == 'bessel':
            pupil.functiontype('annulus', Kxy_max, Kxy_max-2*self.dK)
        elif mtype == 'SIM':
            width=2*self.dK
            pupil.functiontype('delta', width, [Kxy_max-width/2, 0],  [-Kxy_max+width/2, 0])
        elif mtype == 'STED':    
            pupil.functiontype('angular_phase', Kxy_max)
        elif mtype == 'aberrated':
            pupil.functiontype('quartic_phase', Kxy_max*2, 16) 
            #simulates a spherical aberration with phase = kxy^4 
        else:
            raise TypeError("Microscope type '" + mtype + "' not supported")   
        
        indexes= (np.sqrt((kx)**2+(ky)**2))>Kxy_max
        pupil.data[indexes] = 0 # limits the pupil to the NA
        asf = ifftshift(ifft2(fftshift(pupil.data))) 
        projection = pupil.data[:]
        projection.shape = self.N,self.N,1 # reshapes the pupil to be projected on self.values along kz
        self.values = self.values*projection
        
        return pupil.data, asf