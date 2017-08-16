import numpy as np
import cv2
import os

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

    dst = os.path.join(out_dir, sample_name + '_mask.jpg')
    cv2.imwrite(dst, mask)

def load_mask(img_path):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return mask




