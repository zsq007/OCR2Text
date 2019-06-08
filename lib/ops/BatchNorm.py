import tensorflow as tf


def cond_batch_norm(name, axes, inputs, labels=None, n_labels=10):
    # if axes != [0, 1]:
    #     raise Exception('Please use NWC format')
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        num_channels = inputs.get_shape().as_list()[-1]
        with tf.device('/cpu:0'):
            offset_m = tf.get_variable('offsets_by_category', shape=(n_labels, num_channels),
                                       initializer=tf.zeros_initializer())
            scale_m = tf.get_variable('scales_by_category', shape=(n_labels, num_channels),
                                      initializer=tf.ones_initializer())
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)

        if axes == [0, 1]:
            offset = offset[:, None, :]
            scale = scale[:, None, :]

        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
        return result

def cond_mixing_batch_norm(name, axes, inputs, labels=None, n_labels=4):
    if axes != [0, 1]:
        raise Exception('Please use NWC format')
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        num_channels = inputs.get_shape().as_list()[-1]
        with tf.device('/cpu:0'):
            offset_m = tf.get_variable('offsets_by_category', shape=(n_labels, num_channels),
                                       initializer=tf.zeros_initializer())
            scale_m = tf.get_variable('scales_by_category', shape=(n_labels, num_channels),
                                      initializer=tf.ones_initializer())
        offset = tf.matmul(labels, offset_m)[:, None, :]
        scale = tf.matmul(labels, scale_m)[:, None, :]
        # print(inputs.get_shape().as_list())
        # print(offset.get_shape().as_list())
        # print(scale.get_shape().as_list())
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
        return result


def Batchnorm(name, axes, inputs, is_training=True, stats_iter=None, update_moving_stats=True):
    if axes != [0, 1, 2]:
        raise Exception('Please use NHWC format')

    num_channels = inputs.get_shape().as_list()[-1]
    with tf.device('/cpu:0'):
        offset = tf.get_variable('offset', shape=(num_channels,),
                                   initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', shape=(num_channels,),
                                  initializer=tf.ones_initializer())

        moving_mean = tf.get_variable('moving_mean', shape=(num_channels,),
                                   initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('moving_variance', shape=(num_channels,),
                                  initializer=tf.ones_initializer(), trainable=False)

    def _fused_batch_norm_training():
        # Returns normalized batch, empirical mean and unbiased empirical variance
        return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NHWC')

    def _fused_batch_norm_inference():
        # Version which blends in the current item's statistics
        batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
        mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        mean = ((1. / batch_size) * mean) + (((batch_size - 1.) / batch_size) * moving_mean)[None, None, None, :]
        var = ((1. / batch_size) * var) + (((batch_size - 1.) / batch_size) * moving_variance)[None, None, None, :]  # expand dims
        return tf.nn.batch_normalization(inputs, mean, var, offset[None, None, None, :], scale[None, None, None, :], 1e-5), mean, var


    outputs, batch_mean, batch_var = tf.cond(is_training,
                                             _fused_batch_norm_training,
                                             _fused_batch_norm_inference)
    if update_moving_stats:
        no_updates = lambda: outputs

        def _force_updates():
            """Internal function forces updates moving_vars if is_training."""
            float_stats_iter = tf.cast(stats_iter, tf.float32)

            update_moving_mean = tf.assign(moving_mean,
                                           ((float_stats_iter / (float_stats_iter + 1)) * moving_mean) + (
                                                       (1 / (float_stats_iter + 1)) * batch_mean))
            update_moving_variance = tf.assign(moving_variance,
                                               ((float_stats_iter / (float_stats_iter + 1)) * moving_variance) + (
                                                           (1 / (float_stats_iter + 1)) * batch_var))

            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(outputs)

        outputs = tf.cond(is_training, _force_updates, no_updates)

    return outputs