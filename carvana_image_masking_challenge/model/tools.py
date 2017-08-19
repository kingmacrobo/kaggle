import numpy as np
import cv2
import os

from PIL import Image


def length_encoder(mask):
    mask = mask.flatten()

    starts = np.array(mask[:-1] == 0) & np.array(mask[1:0] == 1)
    ends = np.array(mask[:-1] == 1) & np.array(mask[1:0] == 0)
    starts_idx = np.where(starts)[0] + 2
    ends_idx = np.where(ends)[0] + 2

    lengths = ends_idx - starts_idx

    encoded = []
    for i, start in enumerate(starts_idx):
        encoded.append(start)
        encoded.append(lengths[i])

    return encoded


def mask_to_img(mask, out_dir, sample_name):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    dst = os.path.join(out_dir, sample_name + '_mask.gif')
    mask = mask*255
    Image.fromarray(mask).save(dst)


def load_mask(img_path):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return mask

def iou_check(mask_a, mask_b):
    # calculate the iou accuracy
    a = mask_a.flatten()
    b = mask_b.flatten()

    print a.shape, b.shape

    s = np.sum(mask_a)
    e = np.sum(mask_b)

    print np.where(a)
    print np.where(b)

    u = np.sum(a & b)

    accuracy = 2.0 * u / (s + e)

    return accuracy

def main():
    import sys
    from PIL import Image

    a = sys.argv[1]
    b = sys.argv[2]

    img_a = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
    img_b = np.array(Image.open(b))

    print iou_check(img_a, img_b)


if __name__ == '__main__':
    main()
