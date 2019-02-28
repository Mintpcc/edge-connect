import scipy
import random
from scipy.misc import imread
from scipy.misc import imsave
from skimage.color import rgb2gray
from glob import glob
from scipy import ndimage
import os
import numpy as np

files = glob(os.path.join('./data/train_mask_dataset/', '*.png'))
mask_index = random.randint(0, len(files) - 1)
img = imread(files[mask_index])
imsave('mask_in.png', img)
imgh, imgw = img.shape[0:2]
height = width = 256


#side = np.minimum(imgh, imgw)
rh = random.randint(0, imgh - height - 1)
rw = random.randint(0, imgw - width - 1)
#img = img[rh:rh + height, rw:rw + width, ...]

k = random.randint(0, 3)
print(k)
angle = [0, 90, 180, 270]
img = scipy.misc.imrotate(img, angle[k])
imsave('mask_o.png', img)
for i in range(random.randint(0, 10)):
    img = ndimage.binary_dilation(img)

img = (img > 0).astype(np.uint8) * 255

img = scipy.misc.imresize(img, [height, width])
mask = rgb2gray(img)

#mask = scipy.misc.imresize(mask, [height, width])
imsave('mask_out.png', mask)
