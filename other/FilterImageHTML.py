'''
Created on 2 apr 2020
Creates an html page that shows examples of spatial filtering a 2D image in Fourier domain
@author: Andrea Bassi, Politecnico di Milano
'''

import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from ImageType import image
from Function2D import function2D
from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.plotting import output_file, figure
from bokeh.models import Range1d
#from bokeh.models import Select, Slider, RadioGroup, CustomJS
#from bokeh.models import BoxSelectTool, WheelZoomTool, LassoSelectTool

#####################create ImageType with squares
testimage=image('file')   #choose between 'file' and 'rect'
Z=testimage.im
KX, KY = testimage.createKspace()

#####################################################################################
# calculate 2D fft (translated to the center using fftshift. ifftshift is used to remove phase error)
ft=fftshift(fft2(ifftshift(Z)))

filter_function = function2D(KX, KY)
kmax=np.amax(KX)
filter_function.functiontype('square', kmax/4) 
"""
choose between 'circle', 'annulus', 'square', 'gaussian', 'ramp', 'delta'
"""

filt = filter_function.data
#filter ImageType in Fourier space
ft_filtered= ft*filt

##################################################################################
# create filtered ImageType
Z_filtered=ifftshift(ifft2(fftshift(ft_filtered)))

# create filter impulse response
SpreadFunction=ifftshift(ifft2(fftshift(filt)))

def mimage(data, t):
    data=np.flipud(data)
    SC = 1
    p = figure(title=t, tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom")
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.image(image=[data], x=0, y=0, dw=SC, dh=SC, palette="Greys256", level="image")
    p.grid.grid_line_width = 0.1
    p.x_range = Range1d(start=0, end=SC)
    p.y_range = Range1d(start=0, end=SC)
    p.xaxis.visible = None
    p.yaxis.visible = None
    # p.xgrid.grid_line_color = None
    # p.ygrid.grid_line_color = None
    return p


def psd (value):
    epsilon = 1e-9  # constant to avoid log 0 later
    return 10 * np.log10(abs(value + epsilon) ** 2)


s1 = mimage(np.abs(Z),"Original image")
s2 = mimage(np.abs(SpreadFunction),"Impulse response")
s3 = mimage(np.abs(Z_filtered),"Filtered image")

s4 = mimage(psd(ft),"Original image - Power spectral density (log)")
s5 = mimage(psd(filt),"Transfer Function - Power spectral density (log)")
s6 = mimage(psd(ft_filtered),"Filtered image \n Power spectral density (log)")

# make a grid
grid = gridplot([s1, s2, s3, s4, s5, s6], ncols=3, plot_width=300, plot_height=300)#, toolbar_location = None)

output_file("Filter.html", title="Spatial filter")

show(grid)


