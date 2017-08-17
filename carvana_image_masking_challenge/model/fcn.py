import os
import sys
import json
import numpy as np
import tensorflow as tf

import tools

class FCN():
    def __init__(self, datagen, batch_size=64, lr=0.0001, dropout=0.5, model_dir='checkpoints', out_mask_dir= 'out_mask'):
        self.datagen = datagen
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.model_dir = model_dir
        self.out_mask_dir = out_mask_dir
        self.acc_file = os.path.join(self.model_dir, 'accuracy.json')

        print 'batch size: {}, learning reate: {}, dropout: {}\n'.format(self.batch_size, self.lr, self.dropout)

    def conv2d(self, x, filter, scope, activation='relu'):
        with tf.variable_scope(scope):
            w = self.weight_variable(filter)
            b = self.bias_variable([filter[-1]])
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)

            if activation == 'sigmoid':
                return tf.nn.sigmoid(x)
            return tf.nn.relu(x)

    def maxpooling(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

    def weight_variable(self, shape):
        init = tf.truncated_normal(shape, stddev=1)
        return tf.Variable(init)

    def bias_variable(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

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
        net = self.conv2d(drop, [1, 1, 256, 256*256], 'conv9', activation='sigmoid')

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
        loss = tf.reduce_mean(tf.abs(fcn-y))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss, global_step=global_step)


        saver = tf.train.Saver(max_to_keep=3)

        # evaluate fcn
        tf.get_variable_scope().reuse_variables()
        eval_x = tf.placeholder(tf.float32, [1, None, None, 3])
        eval_fcn = self.fcn_net(eval_x, train=False)

        session.run(tf.global_variables_initializer())

        # restore the model
        last_step = 0
        last_acc = 0
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            last_step = int(ckpt.model_checkpoint_path.split('-')[-1])
            saver.restore(session, self.model_dir)
            if os.path.exists(self.acc_file):
                acc_json = json.load(open(self.acc_file, 'r'))
                last_acc = acc_json['accuracy']


        generate_train_batch = self.datagen.generate_batch_train_samples(batch_size=self.batch_size)
        for step in xrange(last_step, 10000000):
            batch_x, batch_y = generate_train_batch.next()
            _, loss_out = session.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})

            if step % 10 == 0:
                print 'step {}, loss {}'.format(step, loss_out)

            if step != 0 and  step % 1000 == 0:
                print 'Evaluate one validate set ... '
                iou_acc = 0
                val_sample_count = self.datagen.get_validate_sample_count()
                validate_samples = self.datagen.generate_validate_samples()
                for _ in xrange(val_sample_count):
                    val_one_x, val_one_y, sample_name = validate_samples.next()
                    val_out = session.run(eval_fcn, feed_dict={eval_x: val_one_x})
                    iou_acc, mask = self.iou_accuracy(val_out, val_one_y)
                    tools.mask_to_img(mask, self.out_mask_dir, sample_name)

                avg_iou_acc = iou_acc/val_sample_count
                print "Validate Set IoU Accuracy: {}".format(avg_iou_acc)

                # save model if get higher accuracy
                if avg_iou_acc > last_acc:
                    last_acc = avg_iou_acc
                    model_path = saver.save(session, self.model_dir)
                    acc_json = {'accuracy': last_acc}
                    json.dump(acc_json, open(self.acc_file, 'w'), indent=4)
                    print 'Get higher accuracy, {}. Save model at {}, Save accuracy at {}'\
                        .format(last_acc, model_path, self.acc_file)


    def eval(self, session):
        # evaluate fcn
        eval_x = tf.placeholder(tf.float32, [None, None, None, 3])
        eval_fcn = self.fcn_net(eval_x, train=False)

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
        summary = np.zeros([height, width, 2])

        # sum the patch results in eval_out
        for i, a in enumerate(eval_out):
            for j, b in enumerate(a):
                # b is one patch of the fcn result
                # convert logistic score to 0, 1 classification
                b = np.reshape(b, [256, 256])
                b = np.round(b).astype(int)
                for h, c in enumerate(b):
                    for w, d in enumerate(c):
                        # d is one pixel classification
                        summary[i+h][j+w][d] += 1

        # choice the max amount one
        summary = np.argmax(summary, axis=2)
        mask = summary

        # calculate the iou accuracy
        s = np.sum(summary)
        e = np.sum(eval_y)
        summary = summary.flatten()
        eval_y = eval_y.flatten()
        u = 0
        for i, p in enumerate(summary):
            if p == 1 and p == eval_y[i]:
                u += 1

        accuracy = 2.0 * u / (s + e)

        return accuracy, mask














