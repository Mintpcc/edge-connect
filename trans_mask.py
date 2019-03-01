import os
from glob import glob
import cv2

path = './data/train_mask_dataset/'
if not os.path.exists(path):
    os.mkdir(path)

mask_list = glob(os.path.join('./data/disocclusion_img_mask', '*.png'))

for i in range(len(mask_list)):
    img = cv2.imread(mask_list[i])
    img = 255 - img
    img_name = mask_list[i].split('/')[-1]
    cv2.imwrite(path + img_name, img)