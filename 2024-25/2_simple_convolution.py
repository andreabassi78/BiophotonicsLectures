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


F = fftshift(fft2(f))
plt.figure()
plt.imshow(np.log(np.abs(F)**2)+0.0000001)
plt.title('Power spectrum of the image (log)')

N = f.shape[0] # N =532

x = y = np.linspace(-1.0, 1.0, N)
X,Y = np.meshgrid(x,y)
w = 0.01
# define g as a Gaussian function
g = np.exp((-X**2-Y**2)/w**2)
plt.figure()
plt.imshow(g)
plt.title('Kernel')

G = fftshift(fft2(g))
plt.figure()
plt.imshow(np.abs(G))
plt.title('Transfer function')

H = F*G

h = ifftshift(ifft2(H))
plt.figure()
plt.imshow(np.abs(h))
plt.title('Filtered image')



plt.show()