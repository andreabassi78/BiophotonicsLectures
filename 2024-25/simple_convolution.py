import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

folder = os.path.dirname(os.path.realpath(__file__))
filename = 'nyc.jpg'

im = mpimg.imread(folder+'\\'+filename)
plt.figure()
plt.imshow(im)
plt.title('original image')

plt.show()