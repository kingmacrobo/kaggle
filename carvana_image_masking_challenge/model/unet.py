import os
import time
import json
import numpy as np
import tensorflow as tf
import cv2

import tools
from layers import conv2d, deconv2d, maxpooling, concat

class UNET():
    def __init__(self, datagen, batch_size=1, lr=0.0005, dropout=0.75, model_dir='checkpoints', out_mask_dir= 'out_mask'):

        self.datagen = datagen
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.model_dir = model_dir
        self.out_mask_dir = out_mask_dir
        self.acc_file = os.path.join(self.model_dir, 'accuracy.json')
        self.loss_log = open('loss_log', 'w')
        self.acc_log = open('acc_log', 'w')

        self.input_size = 1024

        print 'batch size: {}, learning reate: {}, dropout: {}\n'.format(self.batch_size, self.lr, self.dropout)


    def u_net(self, x, layers=5, base_channel=64):
        ds_layers = {}

        # down sample layers
        for layer in range(0, layers-1):
            f_channels = base_channel * (2**layer)
            layer_name = 'ds_{}'.format(layer)
            if layer == 0:
                x = conv2d(x, [3, 3, 3, f_channels], layer_name + '_1')
            else:
                x = conv2d(x, [3, 3, f_channels/2, f_channels], layer_name + '_1')

            x = conv2d(x, [3, 3, f_channels, f_channels], layer_name + '_2')
            ds_layers[layer] = x

            x = maxpooling(x)

        # bottom layer
        f_channels = base_channel * (2**(layers-1))
        x = conv2d(x, [3, 3, f_channels/2, f_channels], 'bottom_1')
        x = conv2d(x, [3, 3, f_channels, f_channels], 'bottom_2')

        # up sample layers
        for layer in range(layers-2, -1, -1):
            f_channels = base_channel * (2**layer)
            layer_name = 'up_{}'.format(layer)
            x = deconv2d(x, [3, 3, f_channels, 2*f_channels], layer_name + '_deconv2d')

            # add the previous down sumple layer to the up sample layer
            x = concat(ds_layers[layer], x)

            x = conv2d(x, [3, 3, 2*f_channels, f_channels], layer_name + '_conv_1')
            x = conv2d(x, [3, 3, f_channels, f_channels], layer_name + '_conv_2')
            x = tf.nn.dropout(x, self.dropout)

        # add 1x1 convolution layer to change channel to 3
        x = conv2d(x, [1, 1, base_channel, 3], 'conv_1x1', activation='no')

        logits = tf.squeeze(x, axis=3)

        return logits

    def train(self, session):
        # train fcn
        x = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])
        y = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size])
        net = self.u_net(x)

        # sigmoid cross entropy loss
        loss_sum = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=net), axis=[1, 2])
        loss = tf.reduce_mean(loss_sum)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        learning_rate = tf.train.exponential_decay(self.lr, global_step,
                                                   3000, 0.95, staircase=True)

        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            loss,
            global_step=global_step,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        saver = tf.train.Saver(max_to_keep=3)

        # evaluate fcn
        tf.get_variable_scope().reuse_variables()
        eval_x = tf.placeholder(tf.float32, [1, self.input_size, self.input_size, 3])
        eval_net = tf.arg_max(tf.nn.softmax(self.u_net(eval_x)), 3)

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
        total_loss = 0
        count = 0
        for step in xrange(last_step + 1, 100000000):
            gd_a = time.time()
            batch_x, batch_y = generate_train_batch.next()
            gd_b = time.time()

            tr_a = time.time()
            _, loss_out = session.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
            tr_b = time.time()

            total_loss += loss_out
            count += 1

            if step % 10 == 0:
                avg_loss = total_loss/count
                print 'global step {}, epoch {}, step {}, loss {}, generate data time: {:.2f} s, step train time: {:.2f} s'\
                    .format(step, step / 4493, step % 4493, avg_loss, gd_b - gd_a, tr_b - tr_a)
                self.loss_log.write('{} {}\n'.format(step, avg_loss))
                total_loss = 0
                count = 0

            if step != 0 and step % 500 == 0:
                model_path = saver.save(session, os.path.join(self.model_dir, 'unet'))
                if os.path.exists(self.acc_file):
                    j_dict = json.load(open(self.acc_file))
                else:
                    j_dict = {'accuracy': 0}

                j_dict['step'] = step
                json.dump(j_dict, open(self.acc_file, 'w'), indent=4)
                print 'Save model at {}'.format(model_path)

            if step != 0 and step % 3000 == 0:
                print 'Evaluate validate set ... '
                iou_acc_total = 0
                val_sample_count = self.datagen.get_validate_sample_count()
                validate_samples = self.datagen.generate_validate_samples()
                for i in xrange(val_sample_count):
                    ed_a = time.time()
                    val_one_x, val_one_y, sample_name = validate_samples.next()
                    ed_b = time.time()

                    ee_a = time.time()
                    val_out = session.run(eval_net, feed_dict={eval_x: val_one_x})
                    iou_acc, mask = self.iou_accuracy(val_out, val_one_y)
                    ee_b = time.time()

                    iou_acc_total += iou_acc

                    if i % 5 == 0:
                        tools.mask_to_img(mask, self.out_mask_dir, sample_name)

                    print '[{}] evaluate {}, accuracy: {:.2f}, load: {:.2f} s, evaluate: {:.2f} s'\
                        .format(i, sample_name, iou_acc, ed_b - ed_a, ee_b - ee_a)

                avg_iou_acc = iou_acc_total/val_sample_count

                self.acc_log.write('{} {}\n'.format(step, avg_iou_acc))
                print "Validate Set IoU Accuracy: {}".format(avg_iou_acc)

                # save model if get higher accuracy
                if avg_iou_acc > last_acc:
                    last_acc = avg_iou_acc
                    model_path = saver.save(session, os.path.join(self.model_dir, 'best'))
                    acc_json = {'accuracy': last_acc, 'step': step}
                    with open(self.acc_file, 'w') as f:
                        json.dump(acc_json, f, indent=4)

                    print 'Get higher accuracy, {}. Save model at {}, Save accuracy at {}'\
                        .format(last_acc, model_path, self.acc_file)


    def eval(self, session):
        # evaluate fcn
        eval_x = tf.placeholder(tf.float32, [None, None, None, 3])
        eval_net = tf.arg_max(tf.nn.softmax(self.u_net(eval_x)), 3)

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
            sample_out = session.run(eval_net, feed_dict={eval_x: sample_x})
            iou_acc, mask = self.iou_accuracy(sample_out, sample_y)
            tools.mask_to_img(mask, self.out_mask_dir, sample_name)

        avg_iou_acc = iou_acc/test_sample_count
        print "Test Set IoU Accuracy: {}".format(avg_iou_acc)

    def iou_accuracy(self, eval_out, eval_y):
        # remove the first dimension since the size is 1
        eval_out = np.squeeze(eval_out, axis=0)

        height, width = eval_y.shape

        eval_out[eval_out==2] = 1

        mask = cv2.resize(eval_out, (width, height), interpolation=cv2.INTER_CUBIC)
        mask = np.round(mask).astype(np.int8)

        # calculate the iou accuracy
        s = np.sum(mask)
        e = np.sum(eval_y)
        u = np.sum(mask & eval_y)

        accuracy = 2.0 * u / (s + e)

        return accuracy, mask
