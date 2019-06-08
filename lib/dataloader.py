import os
import sys
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from functools import partial
from sklearn.model_selection import KFold

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

train_data_dir = os.path.join(basedir, 'Data', 'cell_images', 'training_set')
train_target_file = os.path.join(basedir, 'Data', 'cell_images', 'training_set_values.txt')
test_data_dir = os.path.join(basedir, 'Data', 'cell_images', 'validation_set')
test_target_file = os.path.join(basedir, 'Data', 'cell_images', 'validataion_set_values.txt')

all_allowed_characters = list(map(lambda i: str(i), range(10))) + ['-', '.', ',', '!'] # '!' is the eol signal


def determine_largest_size(path_images):
    max_size = [0, 0]
    for path_image in path_images:
        im = Image.open(path_image)
        for i, s in enumerate(im.size):
            if max_size[i] < s:
                max_size[i] = s
    return max_size


def load_and_preprocess_image(path, max_size):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, max_size)
    image /= 255.0  # normalize to [0,1] range
    return image


def load_ocr_dataset(**kwargs):
    all_train_images = glob.glob(os.path.join(train_data_dir, '*.jpg'))
    all_test_images = glob.glob(os.path.join(train_data_dir, '*.jpg'))
    max_size = determine_largest_size(all_train_images + all_test_images)

    '''prepare the training targets'''
    all_targets = []
    all_ids = []
    with open(train_target_file, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            all_ids.append(line.rstrip().split(';')[0])
            target = line.rstrip().split(';')[-1]
            encode_target = []
            for i, c in enumerate(target):
                encode_target.append(all_allowed_characters.index(c))
            if i < max_size[0] - 1:
                encode_target += [all_allowed_characters.index('!')] * (max_size[0] - 1 - i)
            # this is to ensure all targets are of the same length
            all_targets.append(encode_target)
    all_targets = np.array(all_targets)

    # note that the images returned by glob is unsorted
    all_train_images = [os.path.join(train_data_dir, id) for id in all_ids]

    # print(all_train_images)
    # print(all_targets)

    image_load_func = partial(load_and_preprocess_image, max_size=max_size)
    path_ds = tf.data.Dataset.from_tensor_slices(all_train_images)
    label_ds = tf.data.Dataset.from_tensor_slices(all_targets)
    image_ds = path_ds.map(image_load_func, num_parallel_calls=2)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # print('image shape: ', image_label_ds.output_shapes[0])
    # print('label shape: ', image_label_ds.output_shapes[1])
    # print('types: ', image_label_ds.output_types)
    # print()
    # print(image_label_ds)

    return image_label_ds

if __name__ == "__main__":
    # max_size = determine_largest_size([train_data_dir, test_data_dir])
    # print(max_size)

    load_ocr_dataset()