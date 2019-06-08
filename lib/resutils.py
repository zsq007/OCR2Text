import tensorflow as tf
import functools
import lib.ops.Conv2D, lib.ops.Linear, lib.ops.LSTM


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    """
    For down-sampling - uses average pooling of size 2 and stride 2;
    Originally resnet uses stride 2 in the conv layer for the purpose of down-sampling,
    which is equivalent to max pooling of size 2 and stride 2.
    """
    output = lib.ops.Conv2D.conv2d(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.nn.pool(output, [2, 2], 'AVG', 'SAME', strides=[2, 2])
    # output = tf.add_n([output[:, ::2, :], output[:, 1::2, :]]) / 2.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    # average pooling with size 2 and stride 2
    output = tf.nn.pool(output, [2, 2], 'AVG', 'SAME', strides=[2, 2])
    # output = tf.add_n([output[:, ::2, :], output[:, 1::2, :]]) / 2.
    output = lib.ops.Conv2D.conv2d(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def normalize(name, inputs, is_training_ph, use_bn=True, use_layernorm=False):
    """
    Choosing between ordinary batchnorm and conditional batchnorm;
    Use ordinary batch norm if not conditional;
    Discriminator doesn't use conditional batch norm for the time being.
    """
    with tf.variable_scope(name):
        if use_bn:
            # print(tf.get_variable_scope().reuse)
            return tf.contrib.layers.batch_norm(inputs, fused=True, decay=0.9, is_training=is_training_ph,
                                                scope='BN', reuse=tf.get_variable_scope().reuse,
                                                updates_collections=None)
        elif use_layernorm:
            return tf.contrib.layers.layer_norm(inputs, scope='LN', reuse=tf.get_variable_scope().reuse)
        else:
            return inputs


def resblock(name, input_dim, output_dim, filter_size, inputs, resample, is_training_ph, r=1.0,
             use_bn=True, use_layernorm=False):
    if resample == 'down':
        conv1 = functools.partial(lib.ops.Conv2D.conv2d, input_dim=input_dim, output_dim=input_dim)
        conv2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        shortcut_func = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample is None:
        conv1 = functools.partial(lib.ops.Conv2D.conv2d, input_dim=input_dim, output_dim=output_dim)
        conv2 = functools.partial(lib.ops.Conv2D.conv2d, input_dim=output_dim, output_dim=output_dim)
        shortcut_func = functools.partial(lib.ops.Conv2D.conv2d, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('Wrong choice of resampling.')

    with tf.variable_scope(name):

        if output_dim == input_dim and resample is None:
            shortcut = inputs
        else:
            shortcut = shortcut_func(name='Shortcut', filter_size=1, he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = normalize(name='Norm1', is_training_ph=is_training_ph, inputs=output,
                           use_bn=use_bn, use_layernorm=use_layernorm)
        output = tf.nn.relu(output)
        output = conv1(name='Conv1', filter_size=filter_size, inputs=output)
        output = normalize(name='Norm2', is_training_ph=is_training_ph, inputs=output,
                           use_bn=use_bn, use_layernorm=use_layernorm)
        output = tf.nn.relu(output)
        output = conv2(name='Conv2', filter_size=filter_size, inputs=output)

        return r * output + shortcut


def OptimizedResBlockDisc1(inputs, input_dim, output_dim, resample='down'):
    conv1 = functools.partial(lib.ops.Conv2D.conv2d, input_dim=input_dim, output_dim=output_dim)
    if resample == 'down':
        conv2 = functools.partial(ConvMeanPool, input_dim=output_dim, output_dim=output_dim)
        conv_shortcut = MeanPoolConv
    else:
        conv2 = functools.partial(lib.ops.Conv2D.conv2d, input_dim=output_dim, output_dim=output_dim)
        conv_shortcut = lib.ops.Conv2D.conv2d
    shortcut = conv_shortcut('Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv1('Conv1', filter_size=3, inputs=output)
    output = tf.nn.relu(output)
    output = conv2('Conv2', filter_size=3, inputs=output)
    return shortcut + output
