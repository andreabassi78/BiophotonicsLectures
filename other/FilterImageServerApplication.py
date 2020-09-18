'''
Created on 3 apr 2020
Generates a localhost that runs examples of spatial filtering a 2D image in Fourier domain
@author: Andrea Bassi, Politecnico di Milano
'''

import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from ImageType import image
from Function2D import function2D
#from bokeh.io import show
from bokeh.layouts import gridplot, layout
from bokeh.plotting import figure, curdoc, ColumnDataSource #, output_file,
from bokeh.models import Range1d, Select, Slider


testimage = image('file')
image_in = np.flipud(testimage.im)
KX, KY = testimage.createKspace()

# calculate 2D fft (translated to the center using fftshift. ifftshift is used to remove phase error)
fft_in = fftshift(fft2(ifftshift(image_in)))


def create_filter(filter_type, size):
    filt = function2D(KX, KY)
    #kmax = np.amax(KX)
    filt.functiontype(filter_type, size)
    return filt


def calculate_FFT(fft_in, filt):
    tf = filt.data
    # filter ImageType in Fourier space
    fft_out = fft_in*tf
    # create filtered ImageType
    image_out = ifftshift(ifft2(fftshift(fft_out)))
    # create filter impulse response
    ir = ifftshift(ifft2(fftshift(tf)))
    return ir, image_out, tf, fft_out


filter_function = create_filter('square', 0.2)
ir, image_out, tf, fft_out = calculate_FFT(fft_in, filter_function)

source= {} 

def mimage(data, index, t):
    
    SC = 1
    source[index] = ColumnDataSource({'image': [data]})
    p = figure(title=t, tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom")
    p.x_range.range_padding = p.y_range.range_padding = 0
    
    p.image(image='image', x=0, y=0, dw=SC, dh=SC, source=source[index], palette="Greys256", level="image")
    p.grid.grid_line_width = 0.1
    p.x_range = Range1d(start=0, end=SC)
    p.y_range = Range1d(start=0, end=SC)
    p.xaxis.visible = None
    p.yaxis.visible = None
    # p.xgrid.grid_line_color = None
    # p.ygrid.grid_line_color = None
    return p


def psd(value):
    epsilon = 1e-9  # constant to avoid log 0 later
    return 10 * np.log10(abs(value + epsilon) ** 2)



s0 = mimage(np.abs(image_in),'0',"Original image")
s1 = mimage(np.abs(ir),'1',"Impulse response")
s2 = mimage(np.abs(image_out),'2',"Filtered image")

s3 = mimage(psd(fft_in),'3',"Original image - Power spectral density (log)")
s4 = mimage(psd(tf),'4',"Transfer Function - Power spectral density (log)")
s5 = mimage(psd(fft_out),'5',"Filtered image \n Power spectral density (log)")

select = Select(title="Filter type:", value="square", options=["circle", "square", "gaussian", "ramp", "delta"])
slider = Slider(start=0, end=0.5, value=0.2, step=.01, title="Size")

def update(attrname, old, new):
    filter_type = select.value
    size = slider.value
    filter_new = create_filter(filter_type,size)
    ir_new, image_out_new, tf_new, fft_out_new = calculate_FFT(fft_in, filter_new)

    source['1'].data = {'image': [np.abs(ir_new)]}
    source['2'].data = {'image': [np.abs(image_out_new)]}
    source['4'].data = {'image': [psd(tf_new)]}
    source['5'].data = {'image': [psd(fft_out_new)]}
    

select.on_change('value', update)
slider.on_change('value', update)

# make a grid and the layout
grid = gridplot([s0, s1, s2, s3, s4, s5], ncols=3, plot_width=300, plot_height=300, toolbar_location=None)
lay = layout([[select, slider], grid])

curdoc().add_root(lay)