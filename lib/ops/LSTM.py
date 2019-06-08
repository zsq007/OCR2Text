import tensorflow as tf
import numpy as np
import lib.ops.Linear, lib.ops.BatchNorm, lib.ops.Conv1D
import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def rollout_lstm_policy(name, batch_size, hidden_units, nb_emb, nb_class, latent_encodings, labels, length,
                        inputs, use_cdn=False):
    '''
    Conditional generator via conditional instance normalization or encoding to the initial LSTM state
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell_forward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_cell')

        state = tf.nn.rnn_cell.LSTMStateTuple(c=latent_encodings,
                                              h=tf.zeros(shape=(batch_size, hidden_units)))

        policy_prob = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        tokens = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        new_token = np.zeros((batch_size, nb_emb)).astype(np.float32)

        # unroll inputs
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, tf.shape(inputs)[1])

        def body(i, policy_prob, tokens, state, token):
            cell_output, state = cell_forward(inputs[:, i, :], state)
            if use_cdn:
                prob = tf.nn.softmax(lib.ops.Linear.linear('policy_dist', hidden_units, nb_emb,
                                                           lib.ops.BatchNorm.cond_batch_norm('CDN', [0], cell_output,
                                                                                             labels, nb_class)))
            else:
                prob = tf.nn.softmax(
                    lib.ops.Linear.linear('policy_dist', hidden_units, nb_emb, cell_output))
            policy_prob = policy_prob.write(i, prob)

            new_token = tf.squeeze(tf.one_hot(tf.multinomial(tf.log(prob), 1),
                                              nb_emb, on_value=1.0,
                                              off_value=0.0))  # [batch_size, nb_emb]
            tokens = tokens.write(i, inputs[:, i, :])  # use what we had already in the inputs
            return [tf.add(i, 1), policy_prob, tokens, state, new_token]

        i, policy_prob, tokens, state, new_token = tf.while_loop(while_condition, body,
                                                                 [i, policy_prob, tokens, state, new_token])

        # sample new tokens, for a naive rollout, i starts at 1
        i = tf.shape(inputs)[1]
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, length)

        def body(i, policy_prob, tokens, state, token):  # token: [batch_size, nb_emb]
            tokens = tokens.write(i, token)
            cell_output, state = cell_forward(token, state)
            if use_cdn:
                prob = tf.nn.softmax(lib.ops.Linear.linear('policy_dist', hidden_units, nb_emb,
                                                           lib.ops.BatchNorm.cond_batch_norm('CDN', [0], cell_output,
                                                                                             labels, nb_class)))
            else:
                prob = tf.nn.softmax(lib.ops.Linear.linear('policy_dist', hidden_units, nb_emb, cell_output))
            new_token = tf.squeeze(tf.one_hot(tf.multinomial(tf.log(prob), 1),
                                              nb_emb, on_value=1.0,
                                              off_value=0.0))  # [batch_size, nb_emb]
            policy_prob = policy_prob.write(i, prob)
            return [tf.add(i, 1), policy_prob, tokens, state, new_token]

        i, policy_prob, tokens, state, new_token = tf.while_loop(while_condition, body,
                                                                 [i, policy_prob, tokens, state, new_token])
        tokens = tokens.write(i, new_token)  # add the last generated letter

        policy_prob = tf.transpose(policy_prob.stack(), [1, 0, 2])  # [batch_size, length, nb_emb]
        tokens = tf.transpose(tokens.stack(), [1, 0, 2])  # [batch_size, length, nb_emb]

    return policy_prob, tokens


def lstm_policy(name, batch_size, hidden_units, attention_size, nb_class, latent_encodings, inputs, length,
                keep_prob_ph):
    '''
    A (quite) less efficient implementation, where only an immediate letter is sampled
    given a list of previously samples tokens aka a state.
    '''
    print(inputs.get_shape().as_list())

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell_forward = tf.contrib.rnn.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_cell'),
            output_keep_prob=keep_prob_ph
        )

        state = tf.nn.rnn_cell.LSTMStateTuple(c=latent_encodings, h=tf.zeros(shape=(batch_size, hidden_units)))
        lstm_output = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        policy_prob = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2: tf.less(i, length)

        def body(i, lstm_output, state):
            cell_output, state = cell_forward(inputs[:, i, :], state)
            # cell_output = tf.get_variable('tmp', shape=(256, 100))
            lstm_output = lstm_output.write(i, cell_output)
            return [tf.add(i, 1), lstm_output, state]

        _, lstm_output, state = tf.while_loop(while_condition, body, [i, lstm_output, state])
        lstm_output = tf.transpose(lstm_output.stack(), [1, 0, 2])
        print(lstm_output.get_shape().as_list())

        i = tf.constant(0)
        while_condition_2 = lambda i, _: tf.less(i, length)

        def body(i, policy_prob):
            att_output = attention('policy_summary', attention_size,
                                   lstm_output[:, :i + 1, :])  # [batch_size, attention_size]
            pos_dist = tf.nn.softmax(
                lib.ops.Linear.linear('policy_dist', hidden_units, nb_class, att_output))  # [batch_size, nb_class]
            policy_prob = policy_prob.write(i, pos_dist[:, :])
            return tf.add(i, 1), policy_prob

        _, policy_prob = tf.while_loop(while_condition_2, body, [i, policy_prob])
        policy_prob = tf.transpose(policy_prob.stack(), [1, 0, 2])
        print(policy_prob.get_shape().as_list())

    return policy_prob


def bilstm(name, hidden_units, inputs, length):
    with tf.variable_scope(name):
        cell_forward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_cell')
        cell_backward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='backward_cell')

        state_forward = cell_forward.zero_state(tf.shape(inputs)[0], tf.float32)
        state_backward = cell_backward.zero_state(tf.shape(inputs)[0], tf.float32)

        input_forward = inputs
        input_backward = tf.reverse(inputs, [1])

        output_forward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        output_backward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, length)

        def body(i, output_forward, output_backward, state_forward, state_backward):
            cell_output_forward, state_forward = cell_forward(input_forward[:, i, :], state_forward)
            output_forward = output_forward.write(i, cell_output_forward)
            cell_output_backward, state_backward = cell_backward(input_backward[:, i, :], state_backward)
            output_backward = output_backward.write(i, cell_output_backward)
            return [tf.add(i, 1), output_forward, output_backward, state_forward, state_backward]

        _, output_forward, output_backward, state_forward, state_backward = tf.while_loop(while_condition, body,
                                                                                          [i, output_forward,
                                                                                           output_backward,
                                                                                           state_forward,
                                                                                           state_backward])
        output_forward = tf.transpose(output_forward.stack(), [1, 0, 2])
        output_backward = tf.reverse(tf.transpose(output_backward.stack(), [1, 0, 2]), [1])
        output = tf.concat([output_forward, output_backward], axis=2)

        print(output.get_shape().as_list())
        return output


@deprecated
def legacy_bilstm(name, hidden_units, inputs, keep_prob_ph):
    batch_size, nb_steps, nb_features = inputs.shape.as_list()

    with tf.variable_scope(name):
        cell_forward = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hidden_units, name='forward_cell'),
            output_keep_prob=keep_prob_ph
        )

        cell_backward = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hidden_units, name='backward_cell'),
            output_keep_prob=keep_prob_ph
        )

        state_forward = cell_forward.zero_state(batch_size, tf.float32)
        state_backward = cell_backward.zero_state(batch_size, tf.float32)

        input_forward = inputs
        input_backward = tf.reverse(inputs, [1])

        output_forward = []
        output_backward = []

        # unroll
        for step in range(nb_steps):
            cell_output_forward, state_forward = cell_forward(input_forward[:, step, :], state_forward)
            output_forward.append(cell_output_forward[:, None, :])
            cell_output_backward, state_backward = cell_backward(input_backward[:, step, :], state_backward)
            output_backward.append(cell_output_backward[:, None, :])

        output_forward = tf.concat(output_forward, axis=1)
        output_backward = tf.concat(output_backward, axis=1)
        output = tf.concat([output_forward, output_backward], axis=2)

        return output


def attention(name, attention_size, inputs):
    batch_size, nb_steps, nb_features = inputs.shape.as_list()
    with tf.variable_scope(name):
        context_vec = tf.tanh(
            lib.ops.Linear.linear('Context_Vector', nb_features, attention_size, tf.reshape(inputs, [-1, nb_features])))
        pre_weights_exp = tf.exp(
            tf.reshape(lib.ops.Linear.linear('Attention_weights', attention_size, 1, context_vec),
                       [tf.shape(inputs)[0], -1]))
        weights = pre_weights_exp / tf.reduce_sum(pre_weights_exp, 1)[:, None]
        output = tf.reduce_sum(inputs * weights[:, :, None], 1)
        return output


def self_attention(name, attention_size, inputs, use_conv=False):
    batch_size, nb_steps, nb_features = inputs.shape.as_list()
    with tf.variable_scope(name):
        if use_conv:
            func = functools.partial(lib.ops.Conv1D.conv1d, filter_size=1)
        else:
            func = lib.ops.Linear.linear
        cv_f = func(name='Context_Vector_f', input_dim=nb_features, output_dim=attention_size, inputs=inputs)
        cv_g = func(name='Context_Vector_g', input_dim=nb_features, output_dim=attention_size, inputs=inputs)
        cv_h = func(name='Context_Vector_h', input_dim=nb_features, output_dim=nb_features, inputs=inputs)

        sa_weights = tf.matmul(cv_f, cv_g, transpose_b=True)  # [batch_size, nb_steps, nb_steps]
        sa_weights = tf.nn.softmax(sa_weights, axis=-1)[:, :, :, None]  # [batch_size, nb_steps, nb_steps]

        # tf.transpose(tf.reshape(cv_h, [batch_size, nb_steps, nb_features]), perm=[1, 2, 0])  #[nb_steps, nb_features, batch_size]
        return tf.reduce_sum(sa_weights * tf.stack([cv_h] * nb_steps, axis=1),
                             axis=2)  # [batch_size, nb_steps, nb_features]


if __name__ == "__main__":
    latent_encodings = np.random.randn(256, 128).astype(np.float32)
    sess = tf.Session()
    state = tf.concat(
        [tf.zeros(shape=(256, 1, 4)), tf.one_hot((np.random.rand(256, 127) * 4).astype(np.int32), depth=4)], axis=1)
    policy_dist, tokens = rollout_lstm_policy('test', 256, 128, 4, latent_encodings, 128, inputs=state,
                                              keep_prob_ph=1.0)
    sess.run(tf.global_variables_initializer())
    print('policy_dist shape', sess.run(policy_dist).shape)
    print('generated tokens shape', sess.run(tokens).shape)

    sim = tf.reduce_mean(tf.cast(
        tf.equal(
            tf.to_int32(tf.argmax(state, axis=-1)),
            tf.to_int32(tf.argmax(tokens[:, :-1, :], axis=-1))
        ),
        tf.float32
    ))

    print(sess.run(sim))  # sim is always guaranteed to be 1.0, providing accurate estimate of the policy gradient
    exit()

    lstm_policy('test', hidden_units=10, attention_size=20, nb_class=4,
                latent_encodings=np.random.randn(5, 10).astype(np.float32),
                inputs=tf.placeholder(tf.float32, (5, 5, 4)), keep_prob_ph=1.0)
    print(tf.trainable_variables())
