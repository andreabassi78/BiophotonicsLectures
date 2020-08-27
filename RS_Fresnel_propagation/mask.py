import numpy as np

from diffractio import degrees, um
from diffractio.utils_drawing import draw_several_fields
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY


num_pixels = 512

length = 100 * um
x0 = np.linspace(-length / 2, length / 2, num_pixels)
y0 = np.linspace(-length / 2, length / 2, num_pixels)
wavelength = 0.633 * um

u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
#u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t1.square(r0=(0 * um, 0 * um), size=(2 * um, 2 * um), angle=0 * degrees)
#t1.double_slit(x0=0, size=2 * um, separation=10 * um, angle=0 * degrees)

u2 = u1 * t1
u3 = u2.RS(z=100 * um, new_field=True)

u4 = u2.RS(z=1000 * um, new_field=True)

draw_several_fields((u2,u3,u4), titulos=('mask', '100 um', '1000 um'))

