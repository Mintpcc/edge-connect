import cv2
import numpy as np
import os
from glob import glob

def main():
    path = './test_ini'
    mask_path = './test_m'
    if not os.path.exists('./test_img'):
        os.mkdir('./test_img')
    if not os.path.exists('./test_mask'):
        os.mkdir('./test_mask')
    img_list = glob(os.path.join(path, '*.png'))
    mask_list = glob(os.path.join(mask_path, '*.jpg'))
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        mask = cv2.imread(mask_list[i])
        mask = 255 - mask
        mask = cv2.resize(mask, (256, 256))
        img = np.uint8(np.multiply(img,  1 - mask/255) + mask)
        cv2.imwrite('./test_mask/'+ '%d.jpg'%i, mask)
        cv2.imwrite('./test_img/' + '%d.jpg'%i, img)


if __name__ == "__main__":
    main()