import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

folder = "C:\\Users\\andre\\OneDrive - Politecnico di Milano\\Documenti\\PythonProjects\\Lectures\\FourierTransforms\\"
filename = 'nyc.jpg'


im = mpimg.imread(folder+filename)
plt.figure()
plt.imshow(im)
plt.title('original image')



ft =  np.fft.fftshift(np.fft.fft2(im))
plt.figure()
plt.imshow(np.log(np.abs(ft)+1e-9)) # power spectrum
plt.title('Fourier transform')


kernel = np.zeros_like(im)
N = im.shape[0] # number of pixels along one direction of the original image
x = y = np.linspace(-0.5, 0.5, N)
X, Y = np.meshgrid(x,y)
side = 0.02
indexes = (np.abs(X)<side/2) & (np.abs(Y)<side/2)
kernel[indexes] = 1

plt.figure()
plt.imshow(kernel)
plt.title('kernel')

ft_kernel = np.fft.fftshift(np.fft.fft2(kernel))

filtered_ft = ft*ft_kernel


filtered =  np.fft.ifftshift(np.fft.ifft2(filtered_ft))
plt.figure()
plt.imshow(np.abs(filtered))
plt.title('filtered image')
plt.show()


