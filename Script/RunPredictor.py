import os
import sys
import datetime
import itertools
import numpy as np
import tensorflow as tf

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.dataloader
from lib.logger import CSVLogger
from Model.Predictor_Parallel import Predictor

tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 200, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_bool('use_cross_validation', False, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 200 * FLAGS.nb_gpus  if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
N_EMB = 3 # 3 channels for images
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)]

dataset = lib.dataloader.load_ocr_dataset(use_cross_validation=FLAGS.use_cross_validation)
N_CLASS = len(lib.dataloader.all_allowed_characters)

arch = 0
use_bn = True
use_lstm = True
nb_layers = 4
filter_size = 3
output_dim = 128
learning_rate = 4e-3

HParams = ['arch', 'use_bn', 'use_lstm', 'nb_layers', 'filter_size', 'output_dim', 'learning_rate']
metrics = ['cost', 'char_acc', 'sample_acc']
hp = {}
for param in HParams:
    hp[param] = eval(param)

print('Building model with hyper-parameters\n', hp)

cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if FLAGS.output_dir == '':
    output_dir = os.path.join('output', 'RSCV', cur_time)
else:
    output_dir = os.path.join('output', 'RSCV', cur_time + '-' + FLAGS.output_dir)
os.makedirs(output_dir)

logger = CSVLogger('run.csv', output_dir, HParams + metrics)

splits = dataset['splits']

# build model
model = Predictor(lib.dataloader.max_size, N_EMB, N_CLASS, DEVICES, **hp)
model_dir = os.path.join(output_dir, 'run-%d' % (0))
os.makedirs(model_dir)
cost, char_acc, sample_acc = 0., 0., 0.
all_targets = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(splits_touse):
    fold_dir = os.path.join(model_dir, 'fold-%d' % (fold))
    os.makedirs(fold_dir)
    model.fit(dataset['data'][train_idx], dataset['targets'][train_idx], EPOCHS, BATCH_SIZE, fold_dir)

    test_rmd = dataset['targets'][test_idx].shape[0] % len(DEVICES)
    if test_rmd != 0:
        test_data = dataset['data'][test_idx][:-test_rmd]
        all_targets.append(dataset['targets'][test_idx][:-test_rmd])
    else:
        test_data = dataset['data'][test_idx]
        all_targets.append(dataset['targets'][test_idx])
    all_predictions.append(model.predict(test_data, BATCH_SIZE))

    test_cost, test_acc, test_pears = model.evaluate(dataset['data'][test_idx], dataset['targets'][test_idx],
                                                     BATCH_SIZE)
    cost += test_cost
    acc += test_acc
    pears += test_pears

    model.reset_session()
model.delete()
del model  # release session
draw_scatter_plots(np.concatenate(all_targets, axis=0), np.concatenate(all_predictions, axis=0),
                   CLASSES, model_dir)
met = {}
for metric in metrics:
    met[metric] = eval(metric) / 5
print('Combined fold evaluations', met)
hp.update(met)
logger.update_with_dict(hp)

logger.close()



path_ds = tf.data.Dataset.from_tensor_slices(all_train_images)
    label_ds = tf.data.Dataset.from_tensor_slices(all_targets)
    image_ds = path_ds.map(image_load_func, num_parallel_calls=6)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # print('image shape: ', image_label_ds.output_shapes[0])
    # print('label shape: ', image_label_ds.output_shapes[1])
    # print('types: ', image_label_ds.output_types)
    # print()
    # print(image_label_ds)


BATCH_SIZE = 256

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds