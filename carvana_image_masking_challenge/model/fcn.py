import tensorflow as tf

class FCN():
    def __init__(self, batch_size=128, lr=0.0001, dropout=0.5):
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout

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
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def weight_variable(self, shape):
        init = tf.truncated_normal(shape, stddev=1)
        return tf.Variable(init)

    def bias_variable(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def fcn_net(self, x, y, train=True):
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
        net = self.conv2d(drop, [1, 1, 256, 128*128], 'conv9', activation='sigmoid')

        # squeeze the last two dimension in train
        if train:
            net = tf.squeeze(net, [1, 2], name="squeezed")

        return net

    def train(self):
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        y = tf.placeholder(tf.float32, [None, 128*128])

        fcn = self.fcn_net(x, y)

        # L1 distance loss
        loss = tf.reduce_mean(tf.abs(fcn-y))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss, global_step=global_step)




