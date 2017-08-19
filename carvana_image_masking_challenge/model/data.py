import cv2
import random
import os
import numpy as np

from PIL import Image

class DataGenerator():

    def __init__(self, train_list_file, test_list_file, train_mask_dir, debug_dir=None, train_height=256, train_width=256):

        self.train_heigth = train_height
        self.train_width = train_width

        with open(test_list_file, 'r') as f:
            lines = f.readlines()
            self.test_list = []
            for line in lines:
                self.test_list.append(line.strip())

        self.train_mask_dir = train_mask_dir

        self.split_train_data(train_list_file)

        self.debug = False

        if debug_dir:
            self.debug = True
            self.debug_dir = debug_dir
            self.debug_train = os.path.join(self.debug_dir, 'train')
            self.debug_mask = os.path.join(self.debug_dir, 'mask')

            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)
            if not os.path.exists(self.debug_train):
                os.mkdir(self.debug_train)
            if not os.path.exists(self.debug_mask):
                os.mkdir(self.debug_mask)

        self.load_validate_images()

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

    def load_validate_images(self):
        self.validate_images = []
        self.validate_gt_masks = []

        print 'Loading validate images to memory ...'
        for i, img_path in enumerate(self.validate_list):
            print i, img_path
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            self.validate_images.append(img)

            mask_path = os.path.join(self.train_mask_dir, name.split('.')[0] + '_mask.gif')
            im = Image.open(mask_path)
            mask = np.array(im)
            self.validate_gt_masks.append(mask)
        print 'Load validate images done! ^_^ \n'

    def load_train_images(self):

        self.train_images = []
        self.train_gt_masks = []

        print 'Loading 1000 train images to memory ...'
        self.selected_train = random.sample(self.train_list, 1000)
        for i, img_path in enumerate(self.selected_train):
            print i, img_path
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            self.train_images.append(img)

            mask_path = os.path.join(self.train_mask_dir, name.split('.')[0] + '_mask.gif')
            im = Image.open(mask_path)
            mask = np.array(im)
            self.train_gt_masks.append(mask)

        print 'Random load 1000 train images done! ^_^ \n'

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
        step = 0
        while True:
            # load 1000 train images to memory
            if step % 100000 == 0:
                self.load_train_images()

            batch_images = []
            batch_gt_masks = []
            for i in xrange(batch_size):
                index = random.randint(0, len(self.selected_train)-1)
                image = self.train_images[index]
                gt_mask = self.train_gt_masks[index]

                '''
                # load images from memory (memory cost a lot)
                name = img_path.split('/')[-1]
                image = self.train_images[name]
                gt_mask = self.train_gt_masks[name]
                '''

                c_img, c_gt_mask = self.random_crop(image, gt_mask, self.train_heigth, self.train_width)

                if self.debug:
                    self.save_debug_image(str(i), c_img, c_gt_mask)

                batch_images.append(c_img)
                batch_gt_masks.append(c_gt_mask.flatten())

            batch_images = np.asarray(batch_images)
            batch_gt_masks = np.asarray(batch_gt_masks)

            step += 1

            yield batch_images, batch_gt_masks

    def get_validate_sample_count(self):
        return len(self.validate_list)

    def generate_validate_samples(self):
        for index, img_path in enumerate(self.validate_list):
            name = img_path.split('/')[-1]
            image = self.validate_images[index]
            gt_mask = self.validate_gt_masks[index]

            yield np.array([image]), gt_mask, name

    def get_test_sample_count(self):
        return len(self.test_list)

    def generate_test_samples(self):
        for img_path in self.test_list:
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            yield img, name

    def save_debug_image(self, name, image, mask):
        img_path = os.path.join(self.debug_train, name + '.jpg')
        mask_path = os.path.join(self.debug_mask, name + '.gif')

        mask = mask*255
        cv2.imwrite(img_path, image)
        Image.fromarray(mask).save(mask_path)

