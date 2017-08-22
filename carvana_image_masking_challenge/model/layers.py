import tensorflow as tf

def weight_variable(shape):
    return tf.get_variable("weights",
                           shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape):
    return tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))

# weight shape: [filter_h, filter_w, input_channels, output_channels]
def conv2d(x, w_shape, scope, activation='relu'):
    with tf.variable_scope(scope):
        w = weight_variable(w_shape)
        b = bias_variable([w_shape[-1]])
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        if activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif activation == 'no':
            return x

        return tf.nn.relu(x)

# weight shape: [filter_h, filter_w, output_channels, input_channels]
def deconv2d(x, w_shape, scope, stride=2):
    with tf.variable_scope(scope):
        w = weight_variable(w_shape)
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def maxpooling(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def concat(x1,x2):
    return tf.concat([x1, x2], 3)
