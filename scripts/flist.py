import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/mint/PycharmProjects/edge-connect-master/data/testing_mask_dataset', type=str, help='path to the dataset')
parser.add_argument('--output', default='../data/masks_test.flist', type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.jpg', '.png'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')