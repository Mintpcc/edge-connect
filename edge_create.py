from skimage.feature import canny
from scipy.misc import imread
from scipy.misc import imsave
from skimage.color import rgb2gray
import numpy as np

img1 = imread('celeba_01.png')
img2 = imread('celeba_04.png')

img = img1
img[:, 128:256, :] = img2[:, 128:256, :]
img_gray = rgb2gray(img)
imsave('gray.png', img_gray)

img1[:, 128:256, :] = 255
imsave('in.png', img1)

mask = np.ones([256, 256, 3])
mask[:, 0:128, :] = 0
imsave('mask.png', mask)

edge = canny(img_gray, sigma=2).astype(np.float)
edge[:, 127:129] = 0
imsave('edge.png', edge)
