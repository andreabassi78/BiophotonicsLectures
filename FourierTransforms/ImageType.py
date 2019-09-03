'''
Created on 2 ago 2019

@author: Andrea Bassi, Politecnico di Milano
'''

import numpy as np
import matplotlib.image as mpimg

class image(object):
    '''
    create a class for an image that can be created opening a file or with multiple random rectangles
    
    '''
    
    def __init__(self, imtype):
        
        self.im = self.choosetype(imtype)
        
        
    def choosetype(self, imtype):    
              
        if imtype == 'rect':
            return self.create_random_squares_image()
        
        if imtype == 'file':    
            
            return(self.open_image()) 
            
    def createXY(self, xymin = -1.0, xymax = +1.0, Nsamples = 532):
        self.xymax = xymax
        self.xymin = xymin
        self.Nsamples = Nsamples
        
        self.deltaxy = deltaxy = (xymax-xymin)/Nsamples
        x = y = np.arange(xymin, xymax, deltaxy)
        X, Y = np.meshgrid(x,y)
        return X,Y

    def createKspace(self):
            
        kxymin = -1/(2*self.deltaxy)
        kxymax = 1/(2*self.deltaxy)
        deltakxy = 1/(self.xymax-self.xymin)
        kx = ky = np.arange(kxymin, kxymax, deltakxy)
        KX, KY = np.meshgrid(kx, ky)
        return KX, KY

    def create_random_squares_image(self, halfside = 0.05, N = 30):
        
        X,Y = self.createXY()
        assert X == Y
        im = np.zeros(X.shape)
        xr = np.random.uniform(self.xm, self.ym, N)
        yr = np.random.uniform(self.xm, self.ym, N)
        indexes=im.all()
        for ii in range(0,xr.shape[0]):
            indexes = indexes+((np.abs(self.X - xr[ii]) < halfside) * (np.abs(self.Y - yr[ii]) < halfside))
        im[indexes] = 1
        return im
    
    def open_image(self, filename = 'nyc.jpg' ):
        im = np.float32(mpimg.imread(filename))
        sx, sy = im.shape
        assert sx == sy # the code is currently thought for square images
        self.xymax = xymax = sx
        self.xymin = xymin = 0
        self.Nsamples = Nsamples = sx
        self.deltaxy = (xymax-xymin)/Nsamples    
        return im
         

            