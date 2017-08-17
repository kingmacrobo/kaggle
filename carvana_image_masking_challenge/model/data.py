import cv2
import random
import os
import numpy as np

from PIL import Image

class DataGenerator():

    def __init__(self, train_list_file, test_list_file, train_mask_dir, train_height=256, train_width=256):

        self.train_heigth = train_height
        self.train_width = train_width

        with open(test_list_file, 'r') as f:
            lines = f.readlines()
            self.test_list = []
            for line in lines:
                self.test_list.append(line.strip())

        self.train_mask_dir = train_mask_dir

        self.split_train_data(train_list_file)

        #self.load_images()

    def split_train_data(self, train_list_file, validate_fx=0.1):
        with open(train_list_file, 'r') as f:
            lines = f.readlines()
            data_list = []
            for line in lines:
                data_list.append(line.strip())

            random.seed(40)
            random.shuffle(data_list)
            total = len(data_list)
            validate_size = int(validate_fx * total)
            self.validate_list = data_list[:validate_size]
            self.train_list = data_list[validate_size:]
            random.seed()

            print 'Train samples {}, Validate Samples {}'.format(len(self.train_list), len(self.validate_list))

    def load_images(self):
        self.train_images = {}
        self.train_gt_masks = {}
        self.validate_images = {}
        self.validate_gt_masks = {}

        for i, img_path in enumerate(self.train_list):
            print i, img_path
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            self.train_images[name] = img

            mask_path = os.path.join(self.train_mask_dir, name.split('.')[0] + '_mask.gif')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.train_gt_masks[name] = mask

        for i, img_path in enumerate(self.validate_list):
            print i, img_path
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            self.validate_images[name] = img

            mask_path = os.path.join(self.train_mask_dir, name.split('.')[0] + '_mask.gif')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.validate_gt_masks[name] = mask

        print 'Load train images done! ^_^ \n'

    def load_image_from_file(self, img_path):
        img = cv2.imread(img_path)
        return img

    def load_gt_mask_from_file(self, img_path):
        name = img_path.split('/')[-1].split('.')[0]
        mask_path = os.path.join(self.train_mask_dir, name + '_mask.gif')
        im = Image.open(mask_path)
        mask = np.array(im)
        return mask

    def random_crop(self, image, gt_mask, h, w):
        height, width = image.shape[:-1]

        top = random.randint(0, height - h)
        bot = top + h

        left = random.randint(0, width - w)
        right = left + w

        croped_image = image[top: bot, left: right, : ].copy()
        croped_gt_mask = gt_mask[top: bot, left: right].copy()

        return croped_image, croped_gt_mask

    def generate_batch_train_samples(self, batch_size=64):
        while True:
            batch_images = []
            batch_gt_masks = []
            for _ in xrange(batch_size):
                img_path = random.choice(self.train_list)
                image = self.load_image_from_file(img_path)
                gt_mask = self.load_gt_mask_from_file(img_path)

                '''
                # load images from memory (memory cost a lot)
                name = img_path.split('/')[-1]
                image = self.train_images[name]
                gt_mask = self.train_gt_masks[name]
                '''

                c_img, c_gt_mask = self.random_crop(image, gt_mask, self.train_heigth, self.train_width)
                batch_images.append(c_img)
                batch_gt_masks.append(c_gt_mask.flatten())

            batch_images = np.asarray(batch_images)
            batch_gt_masks = np.asarray(batch_gt_masks)

            yield batch_images, batch_gt_masks

    def get_validate_sample_count(self):
        return len(self.validate_list)

    def generate_validate_samples(self):
        for img_path in self.validate_list:
            name = img_path.split('/')[-1]
            image = self.load_image_from_file(img_path)
            gt_mask = self.load_gt_mask_from_file(img_path)

            yield np.array([image]), gt_mask, name

    def get_test_sample_count(self):
        return len(self.test_list)

    def generate_test_samples(self):
        for img_path in self.test_list:
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            yield img, name


