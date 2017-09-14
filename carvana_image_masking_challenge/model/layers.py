import tensorflow as tf

def weight_variable(shape):
    return tf.get_variable("weights",
                           shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape):
    return tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))

# weight shape: [filter_h, filter_w, input_channels, output_channels]
def conv2d(x, w_shape, scope, activation='relu', bn=False):
    with tf.variable_scope(scope):
        w = weight_variable(w_shape)
        b = bias_variable([w_shape[-1]])
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        #batch normalization
        if bn:
            fc_mean, fc_var = tf.nn.moments(
                x,
                axes=[0, 1, 2]
            )
            out_size = w_shape[-1]
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.00001

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

        if activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif activation == 'no':
            return x

        return tf.nn.relu(x)

# weight shape: [filter_h, filter_w, output_channels, input_channels]
def deconv2d(x, w_shape, ds_shape, scope, stride=2):
    with tf.variable_scope(scope):
        w = weight_variable(w_shape)
        x_shape = tf.shape(x)
        #output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        output_shape = tf.stack([x_shape[0], ds_shape[1], ds_shape[2], x_shape[3]//2])
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