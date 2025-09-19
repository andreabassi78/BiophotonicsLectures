import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import os

folder = os.path.dirname(os.path.realpath(__file__))
filename = 'nyc.jpg'

f = mpimg.imread(os.path.join(folder,filename))

plt.figure()
plt.imshow(f)
plt.title('original image')
plt.show()