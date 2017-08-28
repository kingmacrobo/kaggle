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
    mask = mask * 255
    Image.fromarray(mask).save(dst)

def eval_to_img(eval_out, out_dir, sample_name):
    eval_out = np.squeeze(eval_out, axis=0)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    dst = os.path.join(out_dir, sample_name + '_eval.gif')
    eval_out = eval_out.astype(np.float32)
    mask = eval_out * (255/2)
    mask = cv2.resize(mask, (1918, 1280), interpolation=cv2.INTER_CUBIC)
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

def detect_edge(mask):
    height, width = mask.shape
    for h, row in enumerate(mask):
        for w, p in enumerate(row):
            if p == 1:
                find = False
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        px = x + h
                        py = y + w
                        if px > height-1 or px < 0 or py > width - 1 or py < 0:
                            continue
                        if mask[px][py] == 0:
                            mask[h][w] = 2
                            find = True
                            break
                    if find:
                        break

    return mask

def batch_edge(mask_list_file, out_dir):
    import os
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    mask_list = open(mask_list_file, 'r').readlines()

    for i, img_path in enumerate(mask_list):
        img_path = img_path.strip()
        name = img_path.split('/')[-1]
        out_path = os.path.join(out_dir, name)

        mask = np.array(Image.open(img_path))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_AREA)
        mask = np.round(mask).astype(np.int32)
        mask = detect_edge(mask)
        Image.fromarray(mask).save(out_path)
        print i, out_path

def main():
    import sys

    a = sys.argv[1]
    b = sys.argv[2]
    batch_edge(a, b)

if __name__ == '__main__':
    main()
