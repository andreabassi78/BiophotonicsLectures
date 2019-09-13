'''
Created on 2 ago 2019

@author: Andrea Bassi, Politecnico di Milano
'''
import numpy as np
from xlrd.formula import num2strg
#import matplotlib.image as mpimg

def roundandconvert(num):
    return num2strg(round(num,4))

class function2D(object):
    '''
    Creates a 2-dimensional function (self.data) on a meshgrid X,Y.
    The function can be of different types: circle, annulus, square, gaussian, sine, ramp, delta, 
    angular_phase, quartic_phase.
    self.title is a string containing the filter name and parameters    
    '''

    def __init__(self, X,Y):
        (self.sx, self.sy) = X.shape
        assert X.shape == Y.shape
        assert self.sx == self.sy
        self.X = X
        self.Y = Y
        self.data = np.zeros([self.sx,self.sy])
        self.title = '' # this is a string that is used as title of the function figure, if any 
        self.setcenter(0,0)
        
    def setcenter(self,X0,Y0):
        '''
        centers the function at X0,Y0 (floats)
        '''
        self.X0 = X0    
        self.Y0 = Y0   
      
    def functiontype(self, ftype, arg1, *argv):
        '''
        ftype (string) is the function type
        arg1 (float) is radius for ftype='circle' , halfsize of the side for 'square', waist for 'gaussian'           
        *argv is an optional value (float), for the functions requiring more than 1 argument
        '''    
        if ftype == 'circle':   
            radius=arg1
            indexes= (np.sqrt((self.X-self.X0)**2+(self.Y-self.Y0)**2))<radius
            self.data[indexes] = 1
            self.title = ftype + ': radius = ' + roundandconvert(radius) 
            
        elif ftype == 'annulus':
            ext_radius = arg1 #external radius
            int_radius = ext_radius*3/4 #internal radius, it is ext_radius*3/4 by default, if not specified
            if len(argv) > 0:
                int_radius = argv[0] 
            indexes = ((np.sqrt((self.X-self.X0)**2+(self.Y-self.Y0)**2))<ext_radius) * ((np.sqrt((self.X-self.X0)**2+(self.Y-self.Y0)**2))>int_radius)
            self.data[indexes] = 1
            self.title = ftype + ': external radius = ' + roundandconvert(ext_radius) + 'internal radius = ' + roundandconvert(int_radius)                
                
        elif ftype == 'square':
            halfside=arg1    
            indexes= (np.abs(self.X-self.X0)<halfside) * (np.abs(self.Y-self.Y0)<halfside)
            self.data[indexes] = 1 
            self.title = ftype + ': half-side = ' + roundandconvert(halfside)           
            
        elif ftype == 'gaussian':
            w = arg1   # waist of the gaussian (this would be sqrt(2)*sigma)
            self.data = np.exp((- (self.X-self.X0)**2 - (self.Y-self.Y0)**2)/w**2)
            self.title = ftype + ': waist = ' + roundandconvert(w)  

        elif ftype == 'sine':    
            kx = arg1   # spatial frequency kx 
            ky = 0      # spatial frequency ky, it is 0 by default, if not specified
            if len(argv) > 0:
                ky = argv[0]
            #self.data = ((np.exp(1j*2*np.pi*(kx*(self.X-self.X0) + ky*(self.Y-self.Y0))))) # complex exponential
            self.data = np.sin(2*np.pi*( kx*(self.X-self.X0) + ky*(self.Y-self.Y0))) # sine function
            self.title = ftype + ': kx = ' + roundandconvert(kx) + ', ky = ' + roundandconvert(ky) 

        elif ftype == 'ramp':    
            k = arg1   # slope of the ramp
            self.data = ( k * np.sqrt ((self.X-self.X0)**2 + (self.Y-self.Y0)**2) ) # ramp function
            self.title = ftype + ': k = ' + roundandconvert(k) 
        
        elif ftype == 'delta':
            '''function made of delta in the N points specified in *argv. 
               Example: XXX.functiontype('delta',0.1, [1,0], [5,2])
               will create two delta, of width 0.1, in the two locations [1,0] and [5,2]  
            '''    
            width = arg1   # width of the delta
            for arg in argv:
                X1, Y1 = arg
                indexes= ( np.abs(self.X-self.X0-X1) < width) * ( np.abs(self.Y-self.Y0-Y1)  < width )
                self.data[indexes] = 1 
            self.title = ftype + ': N = ' + roundandconvert(len(argv)) 
        
        elif ftype == 'angular_phase':
            '''simulates a phase increasing angularly to 2pi'''
            radius=arg1
            phase = ( np.arctan2 ((self.Y-self.Y0),(self.X-self.X0)) ) #phase
            phase = phase / (np.amax((phase))-np.amin((phase))) 
            self.data = np.exp(2*np.pi*1j*phase)
            self.title = ftype + ': radius = ' + roundandconvert(radius) 
            
        elif ftype == 'quartic_phase':
            '''simulates a phase increasing with the 4th power of the distance from center'''
            radius = arg1
            c = 1
            if len(argv) > 0:
                c = argv[0]
            phase =  c * (np.sqrt((self.X-self.X0)**2+(self.Y-self.Y0)**2)/radius)**4
            #phase = phase / (np.amax((phase))-np.amin((phase))) 
            self.data = np.exp(2*np.pi*1j*phase)  
            self.title = ftype + ': radius = ' + roundandconvert(radius)     

        else:
            raise TypeError("Function '" + ftype + "' not supported")        
    

        