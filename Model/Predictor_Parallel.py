import numpy as np
import locale
import os
import time
import matplotlib
import sys
import tensorflow as tf

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.resutils import OptimizedResBlockDisc1, resblock, normalize
import lib.ops.LSTM, lib.ops.Linear
import lib.plot


class Predictor:

    def __init__(self, max_size, nb_emb, nb_class, gpu_device_list=['/gpu:0'], **kwargs):
        self.max_size = max_size
        self.nb_emb = nb_emb
        self.nb_class = nb_class
        self.gpu_device_list = gpu_device_list

        # hyperparams
        self.arch = kwargs.get('arch', 0)
        self.use_bn = kwargs.get('use_bn', False)
        self.use_lstm = kwargs.get('use_lstm', True)
        self.nb_layers = kwargs.get('nb_layers', 4)
        self.resample = kwargs.get('resample', None)
        self.filter_size = kwargs.get('filter_size', 3)
        self.residual_connection = kwargs.get('residual_connection', 1.0)
        self.output_dim = kwargs.get('output_dim', 32)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)

        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            for i, device in enumerate(self.gpu_device_list):
                with tf.device(device), tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                    if self.arch == 0:
                        self._build_residual_classifier(i)
                    self._loss('CE' if self.nb_class > 1 else 'MSE', i)
                    self._train(i)
            self._merge()
            self.train_op = self.optimizer.apply_gradients(self.gv)
            self._stats()
            self.saver = tf.train.Saver(max_to_keep=100)
            self.init = tf.global_variables_initializer()
        self._init_session()

    def _placeholders(self):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, *self.max_size, self.nb_emb])
        self.input_splits = tf.split(self.input_ph, len(self.gpu_device_list))

        self.labels = tf.placeholder(tf.int32, shape=[None, self.max_size[0]])
        self.labels_split = tf.split(self.labels, len(self.gpu_device_list))

        self.is_training_ph = tf.placeholder(tf.bool, ())

    def _build_residual_classifier(self, split_idx):
        output = self.input_splits[split_idx]
        for i in range(self.nb_layers):
            if i == 0:
                output = OptimizedResBlockDisc1(output, self.nb_emb, self.output_dim,
                                                resample=self.resample)
            else:
                output = resblock('ResBlock%d' % (i), self.output_dim, self.output_dim, self.filter_size, output,
                                  self.resample, self.is_training_ph, use_bn=self.use_bn, r=self.residual_connection)

        # aggregate conv feature maps
        output = tf.reduce_mean(output, axis=[2])  # more clever attention mechanism for weighting the contribution

        if self.use_lstm:
            output = lib.ops.LSTM.bilstm('BILSTM', self.output_dim, output, self.max_size[0])

        output = lib.ops.Linear.linear('AMOutput', self.output_dim * 2 if self.use_lstm else self.output_dim,
                                       self.nb_class, output)
        # print(output.get_shape().as_list())
        # exit()
        # output = tf.reshape(output, [-1, self.max_size[0], self.nb_class])
        if not hasattr(self, 'output'):
            self.output = [output]
        else:
            self.output += [output]

    def _loss(self, type, split_idx):
        if type == 'CE':
            # compute a more efficient loss?
            # print(self.output[split_idx].get_shape().as_list())
            cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output[split_idx],
                                                               labels=self.labels_split[split_idx]
                                                               ))
            prediction = tf.nn.softmax(self.output[split_idx], axis=-1)
        elif type == 'MSE':
            raise ValueError('MSE is not appropriate in this problem!')
        else:
            raise ValueError('%s doesn\'t supported. Valid options are \'CE\' and \'MSE\'.' % (type))

        if not hasattr(self, 'prediction'):
            self.prediction = [prediction]
        else:
            self.prediction += [prediction]

        # accuracy by comparing the modes
        char_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(self.output[split_idx], axis=-1)),
                    tf.to_int32(tf.argmax(self.labels_split[split_idx], axis=-1))
                ),
                tf.float32
            )
        )

        sample_acc = tf.reduce_mean(
            tf.reduce_prod(
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(self.output[split_idx], axis=-1)),
                        tf.to_int32(tf.argmax(self.labels_split[split_idx], axis=-1))
                    ),
                    tf.float32
                )
                , axis=-1)
        )

        if not hasattr(self, 'cost'):
            self.cost, self.char_acc, self.sample_acc = [cost], [char_acc], [sample_acc]
        else:
            self.cost += [cost]
            self.char_acc += [char_acc]
            self.sample_acc += [sample_acc]

    def _train(self, split_idx):
        gv = self.optimizer.compute_gradients(self.cost[split_idx])
        if not hasattr(self, 'gv'):
            self.gv = [gv]
        else:
            self.gv += [gv]

    def _stats(self):
        # show all trainable weights
        for name, grads_and_vars in [('Predictor arch %d' % (self.arch), self.gv)]:
            print("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print("\t{} ({})".format(v.name, shape_str))
            print("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))

    def _merge(self):
        # output, prediction, cost, acc, pears, gv
        self.output = tf.concat(self.output, axis=0)
        self.prediction = tf.concat(self.prediction, axis=0)

        self.cost = tf.add_n(self.cost) / len(self.gpu_device_list)
        self.char_acc = tf.add_n(self.char_acc) / len(self.gpu_device_list)
        self.sample_acc = tf.add_n(self.sample_acc) / len(self.gpu_device_list)

        self.gv = self._average_gradients(self.gv)

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

    def reset_session(self):
        del self.saver
        with self.g.as_default():
            self.saver = tf.train.Saver(max_to_keep=100)
        self.sess.run(self.init)
        lib.plot.reset()

    def fit(self, X, y, epochs, batch_size, output_dir):
        checkpoints_dir = os.path.join(output_dir, 'checkpoints/')
        os.makedirs(checkpoints_dir)

        # split validation set
        dev_data = X[:int(len(X) * 0.1)]
        dev_targets = y[:int(len(X) * 0.1)]

        # trim development set, batch size should be a multiple of len(self.gpu_device_list)
        dev_rmd = dev_data.shape[0] % len(self.gpu_device_list)
        if dev_rmd != 0:
            dev_data = dev_data[:-dev_rmd]
            dev_targets = dev_targets[:-dev_rmd]

        X = X[int(len(X) * 0.1):]
        y = y[int(len(y) * 0.1):]
        size_train = len(X)
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        best_dev_cost = np.inf
        lib.plot.set_output_dir(output_dir)
        for epoch in range(epochs):
            permute = np.random.permutation(np.arange(size_train))
            train_data = X[permute]
            train_targets = y[permute]

            # trim
            train_rmd = train_data.shape[0] % len(self.gpu_device_list)
            if train_rmd != 0:
                train_data = train_data[:-train_rmd]
                train_targets = train_targets[:-train_rmd]

            start_time = time.time()
            for i in range(iters_per_epoch):
                _data, _labels = train_data[i * batch_size: (i + 1) * batch_size], \
                                 train_targets[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op,
                              feed_dict={self.input_ph: _data,
                                         self.labels: _labels,
                                         self.is_training_ph: True}
                              )

            train_cost, train_char_acc, train_sample_acc = self.evaluate(train_data, train_targets, batch_size)
            lib.plot.plot('train_cost', train_cost)
            lib.plot.plot('train_char_acc', train_char_acc)
            lib.plot.plot('train_sample_acc', train_sample_acc)

            dev_cost, dev_char_acc, dev_sample_acc = self.evaluate(dev_data, dev_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost)
            lib.plot.plot('dev_char_acc', dev_char_acc)
            lib.plot.plot('dev_sample_acc', dev_sample_acc)

            lib.plot.flush()
            lib.plot.tick()

            if dev_cost < best_dev_cost:
                best_dev_cost = dev_cost
                save_path = self.saver.save(self.sess, checkpoints_dir, global_step=epoch)
                print('Validation cost improved. Saved to path %s\n' % (save_path), flush=True)
            else:
                print('\n', flush=True)

        print('Loading best weights %s' % (save_path), flush=True)
        self.saver.restore(self.sess, save_path)

    def evaluate(self, X, y, batch_size):
        iters_per_epoch = len(X) // batch_size + (0 if len(X) % batch_size == 0 else 1)
        all_cost, all_char_acc, all_sample_acc = 0., 0., 0.,
        for i in range(iters_per_epoch):
            _data, _labels = X[i * batch_size: (i + 1) * batch_size], \
                             y[i * batch_size: (i + 1) * batch_size]
            _cost, _char_acc, _sample_acc \
                = self.sess.run([self.cost, self.char_acc, self.sample_acc],
                                feed_dict={self.input_ph: _data,
                                           self.labels: _labels,
                                           self.is_training_ph: False}
                                )
            all_cost += _cost * _data.shape[0]
            all_char_acc += _char_acc * _data.shape[0]
            all_sample_acc += _sample_acc * _data.shape[0]
        return all_cost / len(X), all_char_acc / len(X), all_sample_acc / len(X)

    def predict(self, X, batch_size):
        all_predictions = []
        iters_per_epoch = len(X) // batch_size + (0 if len(X) % batch_size == 0 else 1)
        for i in range(iters_per_epoch):
            _data = X[i * batch_size: (i + 1) * batch_size]
            _prediction = self.sess.run(self.prediction,
                                        {self.input_ph: _data,
                                         self.is_training_ph: False})
            all_predictions.append(_prediction)
        return np.concatenate(all_predictions, axis=0)

    def delete(self):
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
