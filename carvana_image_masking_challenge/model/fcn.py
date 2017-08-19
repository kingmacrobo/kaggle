import os
import time
import json
import numpy as np
import tensorflow as tf

import tools

class FCN():
    def __init__(self, datagen, batch_size=64, lr=0.001, dropout=0.5, model_dir='checkpoints', out_mask_dir= 'out_mask'):
        self.datagen = datagen
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.model_dir = model_dir
        self.out_mask_dir = out_mask_dir
        self.acc_file = os.path.join(self.model_dir, 'accuracy.json')
        self.loss_log = open('loss_log', 'w')
        self.acc_log = open('acc_log', 'w')

        print 'batch size: {}, learning reate: {}, dropout: {}\n'.format(self.batch_size, self.lr, self.dropout)

    def conv2d(self, x, filter, scope, activation='relu'):
        with tf.variable_scope(scope):
            w = self.weight_variable(filter)
            b = self.bias_variable([filter[-1]])
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)

            if activation == 'sigmoid':
                return tf.nn.sigmoid(x)
            elif activation == 'no':
                return x

            return tf.nn.relu(x)

    def maxpooling(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def weight_variable(self, shape):
        return tf.get_variable("weights",
                               shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

    def bias_variable(self, shape):
        return tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))

    def fcn_net(self, x, train=True):
        conv1 = self.conv2d(x, [3, 3, 3, 32], 'conv1')
        maxp1 = self.maxpooling(conv1)

        conv2 = self.conv2d(maxp1, [3, 3, 32, 32], 'conv2')
        maxp2 = self.maxpooling(conv2)

        conv3 = self.conv2d(maxp2, [3, 3, 32, 64], 'conv3')
        maxp3 = self.maxpooling(conv3)

        conv4 = self.conv2d(maxp3, [3, 3, 64, 64], 'conv4')
        maxp4 = self.maxpooling(conv4)

        conv5 = self.conv2d(maxp4, [3, 3, 64, 128], 'conv5')
        maxp5 = self.maxpooling(conv5)

        conv6 = self.conv2d(maxp5, [3, 3, 128, 128], 'conv6')
        maxp6 = self.maxpooling(conv6)

        conv7 = self.conv2d(maxp6, [3, 3, 128, 256], 'conv7')
        maxp7 = self.maxpooling(conv7)

        conv8 = self.conv2d(maxp7, [3, 3, 256, 256], 'conv8')
        maxp8 = self.maxpooling(conv8)

        drop = tf.nn.dropout(maxp8, self.dropout)

        # 1x1 convolution + sigmoid activation
        net = self.conv2d(drop, [1, 1, 256, 256*256], 'conv9', activation='no')

        # squeeze the last two dimension in train
        if train:
            net = tf.squeeze(net, [1, 2], name="squeezed")

        return net

    def train(self, session):
        # train fcn
        x = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
        y = tf.placeholder(tf.float32, [self.batch_size, 256*256])
        fcn = self.fcn_net(x)

        # L1 distance loss
        #loss = tf.reduce_mean(tf.abs(tf.sigmoid(fcn)-y))

        # sigmoid cross entropy loss
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=fcn))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss, global_step=global_step)


        saver = tf.train.Saver(max_to_keep=3)

        # evaluate fcn
        tf.get_variable_scope().reuse_variables()
        eval_x = tf.placeholder(tf.float32, [1, None, None, 3])
        eval_fcn = tf.nn.sigmoid(self.fcn_net(eval_x, train=False))

        session.run(tf.global_variables_initializer())

        # restore the model
        last_step = -1
        last_acc = 0
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            if os.path.exists(self.acc_file):
                acc_json = json.load(open(self.acc_file, 'r'))
                last_acc = acc_json['accuracy']
                last_step = acc_json['step']
            print 'Model restored from {}, last accuracy: {}, last step: {}'\
                .format(ckpt.model_checkpoint_path, last_acc, last_step)


        generate_train_batch = self.datagen.generate_batch_train_samples(batch_size=self.batch_size)
        for step in xrange(last_step + 1, 100000000):
            gd_a = time.time()
            batch_x, batch_y = generate_train_batch.next()
            gd_b = time.time()

            tr_a = time.time()
            _, loss_out = session.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
            tr_b = time.time()

            if step % 10 == 0:
                print 'step {}, loss {}, generate data time: {:.2f} s, step train time: {:.2f} s'\
                    .format(step, loss_out, gd_b - gd_a, tr_b - tr_a)
                self.loss_log.write('{} {}\n'.format(step, loss_out))

            if step % 1000 == 0:
                print 'Evaluate validate set ... '
                iou_acc_total = 0
                val_sample_count = self.datagen.get_validate_sample_count()
                validate_samples = self.datagen.generate_validate_samples()
                val_sample_count = 10
                for i in xrange(val_sample_count):
                    ed_a = time.time()
                    val_one_x, val_one_y, sample_name = validate_samples.next()
                    ed_b = time.time()

                    ee_a = time.time()
                    val_out = session.run(eval_fcn, feed_dict={eval_x: val_one_x})
                    iou_acc, mask = self.iou_accuracy(val_out, val_one_y)
                    ee_b = time.time()

                    iou_acc_total += iou_acc

                    ew_a = time.time()
                    tools.mask_to_img(mask, self.out_mask_dir, sample_name)
                    ew_b = time.time()

                    print '[{}] evaluate {}, accuracy: {:.2f}, load: {:.2f} s, evaluate: {:.2f} s, write: {:.2f} s'\
                        .format(i, sample_name, iou_acc, ed_b - ed_a, ee_b - ee_a, ew_b - ew_a)

                avg_iou_acc = iou_acc_total/val_sample_count

                self.acc_log.write('{} {}\n'.format(step, avg_iou_acc))
                print "Validate Set IoU Accuracy: {}".format(avg_iou_acc)

                # save model if get higher accuracy
                if avg_iou_acc > last_acc:
                    last_acc = avg_iou_acc
                    model_path = saver.save(session, os.path.join(self.model_dir, 'fcn'))
                    acc_json = {'accuracy': last_acc, 'step': step}
                    with open(self.acc_file, 'w') as f:
                        json.dump(acc_json, f, indent=4)

                    print 'Get higher accuracy, {}. Save model at {}, Save accuracy at {}'\
                        .format(last_acc, model_path, self.acc_file)


    def eval(self, session):
        # evaluate fcn
        eval_x = tf.placeholder(tf.float32, [None, None, None, 3])
        eval_fcn = tf.nn.sigmoid(self.fcn_net(eval_x, train=False))

        # load the model
        saver = tf.train.Saver(max_to_keep=3)
        session.run(tf.global_variables_initializer())

        # restore the model
        if not os.path.exist(self.model_dir):
            print 'model dir not found'
            return

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, self.model_dir)
        else:
            print 'restore model failure'
            return

        iou_acc = 0
        test_sample_count = self.datagen.get_test_sample_count()
        test_samples = self.datagen.generate_test_samples()
        for _ in xrange(test_sample_count):
            sample_x, sample_y, sample_name = test_samples.next()
            sample_out = session.run(eval_fcn, feed_dict={eval_x: sample_x})
            iou_acc, mask = self.iou_accuracy(sample_out, sample_y)
            tools.mask_to_img(mask, self.out_mask_dir, sample_name)

        avg_iou_acc = iou_acc/test_sample_count
        print "Test Set IoU Accuracy: {}".format(avg_iou_acc)

    def iou_accuracy(self, eval_out, eval_y):
        # remove the first dimension since the size is 1
        eval_out = np.squeeze(eval_out, axis=0)

        height, width = eval_y.shape
        mask = np.zeros([height, width])

        # sum the patch results in eval_out
        for i, a in enumerate(eval_out):
            for j, b in enumerate(a):
                # b is one patch of the fcn result
                # convert logistic score to 0, 1 classification
                b = np.reshape(b, [256, 256])
                b = np.round(b).astype(np.int8)
                for h, c in enumerate(b):
                    for w, d in enumerate(c):
                        # d is one pixel classification
                        if i*256+h < 1280 and j*256+w < 1918:
                            mask[i*256+h][j*256+w] = d

        # calculate the iou accuracy
        mask = mask.astype(np.int8)
        s = np.sum(mask)
        e = np.sum(eval_y)
        u = np.sum(mask.flatten() & eval_y.flatten())

        accuracy = 2.0 * u / (s + e)

        return accuracy, mask
