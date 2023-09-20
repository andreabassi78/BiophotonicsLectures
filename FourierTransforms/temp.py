import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

N = 512

kernel = np.zeros([N,N]) # this creates an zeros matrix with 512X512 elements
x = y = np.linspace(-0.5, 0.5, N)
X, Y = np.meshgrid(x,y)
side = 0.05
indexes = (np.abs(X)<side/2) & (np.abs(Y)<side/2)
kernel[indexes] = 1

plt.figure()
plt.imshow(kernel)
#plt.gray()
plt.title('kernel')


# open the image
folder = "C:\\Users\\andre\\OneDrive - Politecnico di Milano\\Documenti\\PythonProjects\\Lectures\\FourierTransforms\\"
filename = 'nyc.jpg'
im = mpimg.imread(folder+filename)

plt.figure()
plt.imshow(im)
plt.title('original image')


ft =  np.fft.fftshift(np.fft.fft2(im))
plt.figure()
plt.imshow(np.log(np.abs(ft)**2+1e-9)) # power spectrum
plt.title('Fourier transform')
plt.show()