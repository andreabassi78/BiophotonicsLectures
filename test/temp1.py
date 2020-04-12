'''
Created on 3 apr 2020
Generates a localhost that runs examples of spatial filtering a 2D image in Fourier domain
@author: Andrea Bassi, Politecnico di Milano
'''

import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from FourierTransforms.ImageType import image
from FourierTransforms.Function2D import function2D
from bokeh.io import show
from bokeh.layouts import gridplot, layout
from bokeh.plotting import output_file, figure, curdoc
from bokeh.models import Range1d
from bokeh.models import Select, Slider


def create_filter(filter_type):
    testimage = image('file')
    Z = testimage.im
    KX, KY = testimage.createKspace()
    f_function = function2D(KX, KY)
    kmax = np.amax(KX)
    f_function.functiontype(filter_type, kmax/4)
    return np.abs(f_function.data)

Z = create_filter('square')
SC=1

p = figure(title='temp', tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom")
p.x_range.range_padding = p.y_range.range_padding = 0
p.image(image=[Z], x=0, y=0, dw=SC, dh=SC, palette="Spectral11", level="image")
p.grid.grid_line_width = 0.1
p.x_range = Range1d(start=0, end=SC)
p.y_range = Range1d(start=0, end=SC)
p.xaxis.visible = None
p.yaxis.visible = None
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

def slider_update(attrname, old, new):
    print(new)


def select_update(attrname, old, new):
    filter_type = new
    print(filter_type)
    filter_function = create_filter(filter_type, KX, KY)
    SpreadFunction, Z_filtered, filt, ft_filtered = calculate_FFT(ft, filter_function)


select = Select(title="Filter type:", value="square", options=["circle", "square", "gaussian", "ramp", "delta"])
slider = Slider(start=0, end=10, value=1, step=.1, title="Size")

select.on_change('value', select_update)
slider.on_change('value', slider_update)

# make a grid and the layout
lay = layout([[select, slider], p])

#output_file("temp.html", title="Spatial filter")

#show(lay)

curdoc().add_root(lay)


