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

def dice_coe(output, target, loss_type='sorensen', axis=[1,2,3], smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice