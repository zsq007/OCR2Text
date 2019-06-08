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

dataset_dir = os.path.join(basedir, 'Data', 'cell_images', 'training_set')
dataset_target_file = os.path.join(basedir, 'Data', 'cell_images', 'training_set_values.txt')
expr_data_dir = os.path.join(basedir, 'Data', 'cell_images', 'validation_set')
expr_target_file = os.path.join(basedir, 'Data', 'cell_images', 'validation_set_values.txt')

all_allowed_characters = list(map(lambda i: str(i), range(10))) + ['-', '.', ',', '!'] # '!' is the eol signal
max_size = None
image_load_func = None


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
    # splits the dataset into 10 fold cross validation, or a held-out train-test split
    use_cross_validation = kwargs.get('use_cross_validation', False)

    if kwargs.get('seed') is not None:
        print('Setting seed to %d' % (kwargs.get('seed')))
        np.random.seed(kwargs.get('seed'))

    # to estimate the largest picture width and height, for downstream padding
    # we need uniform length of the input to enable batch optimziation
    all_labeled_images = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    all_expr_images = glob.glob(os.path.join(expr_data_dir, '*.jpg'))
    global max_size, image_load_func
    max_size = determine_largest_size(all_labeled_images + all_expr_images)
    image_load_func = partial(load_and_preprocess_image, max_size=max_size)

    # load all training targets and ids
    all_targets = []
    all_ids = []
    with open(dataset_target_file, 'r') as file:
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
    all_ids = np.array(all_ids)

    # sort the images with the order of the targets
    all_labeled_images = np.array([os.path.join(dataset_dir, id) for id in all_ids])

    # initial shuffle
    total_size = len(all_labeled_images)
    permute = np.random.permutation(np.arange(total_size, dtype=np.int32))
    all_ids = all_ids[permute]
    all_labeled_images = all_labeled_images[permute]
    all_targets = all_targets[permute]

    if use_cross_validation:
        kf = KFold(n_splits=10)
        splits = kf.split(all_labeled_images)
        return {
            'all_ids': all_ids,
            'all_images': all_labeled_images,
            'all_targets': all_targets,
            'splits': splits
        }
    else:
        test_ids = all_ids[-int(total_size * 0.1):]
        test_images = all_labeled_images[-int(total_size * 0.1):]
        test_targets = all_targets[-int(total_size * 0.1):]

        ids = all_ids[:-int(total_size * 0.1)]
        images = all_labeled_images[:-int(total_size * 0.1)]
        targets = all_targets[:-int(total_size * 0.1)]

        print('dataset size %d\ntraining set %d\ntest set %d' % (
            total_size, len(images), len(test_images)))

        return {
            'train_ids': ids,
            'train_images': images,
            'train_targets': targets,
            'test_ids': test_ids,
            'test_images': test_images,
            'test_targets': test_targets
        }


def load_expr_data():
    '''prepare the training targets'''
    all_ids = []
    with open(expr_target_file, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            all_ids.append(line.rstrip().split(';')[0])
    all_expr_images = [os.path.join(expr_data_dir, id) for id in all_ids]
    return all_expr_images, all_ids


if __name__ == "__main__":
    dataset = load_ocr_dataset(use_cross_validation=False)
    print(dataset['train_ids'])
    print(dataset['train_images'])
    print(dataset['train_targets'])