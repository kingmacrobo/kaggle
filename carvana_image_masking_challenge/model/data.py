import cv2
import random
import os
import numpy as np

from PIL import Image

class DataGenerator():

    def __init__(self, train_list_file, test_list_file, train_mask_dir, train_mask_edge_dir, debug_dir=None, input_size=1024):

        self.input_size = input_size

        with open(test_list_file, 'r') as f:
            lines = f.readlines()
            self.test_list = []
            for line in lines:
                self.test_list.append(line.strip())

        self.train_mask_dir = train_mask_dir

        self.train_mask_edge_dir = train_mask_edge_dir

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
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            self.validate_images.append(img)

            mask_path = os.path.join(self.train_mask_dir, name.split('.')[0] + '_mask.gif')
            im = Image.open(mask_path)
            mask = np.array(im)
            self.validate_gt_masks.append(mask)
        print 'Load validate images done! ^_^ \n'

    def load_image_from_file(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = img/255.0
        return img

    def load_gt_mask_from_file(self, img_path):
        name = img_path.split('/')[-1].split('.')[0]
        mask_path = os.path.join(self.train_mask_edge_dir, name + '_mask.gif')
        im = Image.open(mask_path)
        mask = np.array(im)
        mask = mask.astype(np.int32)
        return mask

    def load_train_images(self):

        self.train_images = []
        self.train_gt_masks = []

        print 'Loading 1000 train images to memory ...'
        self.selected_train = random.sample(self.train_list, 1000)
        for i, img_path in enumerate(self.selected_train):
            print i, img_path
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            self.train_images.append(img)

            mask_path = os.path.join(self.train_mask_edge_dir, name.split('.')[0] + '_mask.gif')
            im = Image.open(mask_path)
            mask = np.array(im)
            mask = mask.astype(np.int32)
            self.train_gt_masks.append(mask)

        print 'Random load 1000 train images done! ^_^ \n'

    def generate_batch_train_samples(self, batch_size):
        index = 0
        while True:
            batch_images = []
            batch_gt_masks = []
            for i in range(batch_size):
                img_path = self.train_list[index % len(self.train_list)]
                img = self.load_image_from_file(img_path)
                gt_mask = self.load_gt_mask_from_file(img_path)
                index += 1

                if self.debug:
                    self.save_debug_image(str(i), img, gt_mask)

                batch_images.append(img)
                batch_gt_masks.append(gt_mask)

            batch_images = np.asarray(batch_images)
            batch_gt_masks = np.asarray(batch_gt_masks)

            yield batch_images, batch_gt_masks

    def get_validate_sample_count(self):
        return len(self.validate_list)

    def generate_validate_samples(self):
        for index, img_path in enumerate(self.validate_list):
            name = img_path.split('/')[-1]
            image = self.validate_images[index]/255.0
            gt_mask = self.validate_gt_masks[index]

            yield np.array([image]), gt_mask, name

    def get_test_sample_count(self):
        return len(self.test_list)

    def generate_test_samples(self):
        for img_path in self.test_list:
            name = img_path.split('/')[-1]
            img = cv2.imread(img_path)/255.0
            yield img, name


    def save_debug_image(self, name, image, mask):
        img_path = os.path.join(self.debug_train, name + '.jpg')
        mask_path = os.path.join(self.debug_mask, name + '.gif')

        mask = mask * 255
        image = image * 255
        cv2.imwrite(img_path, image)
        Image.fromarray(mask).save(mask_path)

