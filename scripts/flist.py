import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/mint/PycharmProjects/dataset/ffhq-256', type=str, help='path to the dataset')
parser.add_argument('--output', default='../data/ffhq_', type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.jpg', '.png'}

for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    k = np.random.permutation(len(files))
    images = []
    for i in range(int(0.85 * len(files))):
        if os.path.splitext(files[k[i]])[1] in ext:
            images.append(os.path.join(root, files[k[i]]))
    images = sorted(images)
    np.savetxt(args.output + 'train.flist', images, fmt='%s')

    tmp = int(0.85 * len(files))
    images = []
    for i in range(int(0.05 * len(files))):
        if os.path.splitext(files[k[i + tmp]])[1] in ext:
            images.append(os.path.join(root, files[k[i + tmp]]))
    images = sorted(images)
    np.savetxt(args.output + 'val.flist', images, fmt='%s')

    tmp = int(0.9 * len(files))
    images = []
    for i in range(int(0.1 * len(files))):
        if os.path.splitext(files[k[i + tmp]])[1] in ext:
            images.append(os.path.join(root, files[k[i + tmp]]))
    images = sorted(images)
    np.savetxt(args.output + 'test.flist', images, fmt='%s')