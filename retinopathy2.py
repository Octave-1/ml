# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os
import sys
import random

print(tf.__version__)

# define this for later
AUTOTUNE = tf.data.AUTOTUNE

if len(sys.argv) > 1:
    arg = sys.argv[1]
    first_run = True if (arg == '--first_run') else False
else:
    first_run = False

# define parameters for the loader
batch_size = 32
img_height = 512
img_width = 768
epochs = 1
num_classes = 5
class_names = np.array(['0', '1', '2', '3', '4'])
sample_size = 5000
data_dir = os.getcwd() + '/../datasets/retinopathy/train_images_processed/'
data_dir_test = os.getcwd() + '/../datasets/retinopathy/test_images_processed/'

# create a generator to augment dataset
rng = tf.random.Generator.from_seed(123, alg='philox')

# move images into folders according to level
# train labels
path = os.path.join(os.getcwd(), '../datasets/retinopathy/train_images_processed/')
train_labels = pd.read_csv(os.path.join(path, '../', 'trainLabels.csv'), dtype='string')

# remove one file which could not be processed
train_labels = train_labels.loc[train_labels.image != '492_right']

if first_run:
    for level in class_names:
        labels = train_labels.loc[train_labels.level == level]
        os.mkdir(os.path.join(path, level))

        for image in labels['image']:
            os.rename(path + image + '.jpeg', path + level + "/" + image + '.jpeg')


# (1) randomly sample 10% of the files to get the validation set
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]
files_val = random.sample(files, int(0.1*len(files)))
files_test = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir_test + '/')) for f in fn]

# (2) the get the list of files for each class, removing the validation files
# will use the weights vector in the next step
list_ds = np.array([])
for level in class_names:
    files_train = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + level)) for f in fn]
    files_train = [elem for elem in files_train if elem not in files_val]

    ids = np.arange(len(files_train))
    choices = np.random.choice(ids, sample_size)
    files_train = np.asarray(files_train)[choices]
    list_ds = np.append(list_ds, files_train)


def custom_kappa_metric_ohe(y_true, y_pred):
    # get sparse labels
    y_pred = tf.math.argmax(y_pred, axis=1)
    y_true = tf.math.argmax(y_true, axis=1)

    metric = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=True)
    metric.update_state(y_true, y_pred)
    result = metric.result()

    return result


def load_image(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)

    return img


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    # get the id of the image
    img_id = parts[-1]

    # determine the label
    x = tf.strings.to_number(parts[-2], tf.int32)
    # y = tf.strings.to_number(class_names, tf.int32)

    # The second to last is the class-directory
    # ordinal = tf.math.greater_equal(x, y)

    # Integer encode the label
    return tf.cast(tf.one_hot(x, num_classes), tf.int32), img_id


def process_path(file_path):
    label = get_label(file_path)
    img = load_image(file_path)
    return img, label[0]


def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, img_height, img_width)
    image = (image / 255.0)
    return image, label


def augment(image_label, seed):
    img, label = image_label
    img, label = resize_and_rescale(img, label)

    # Make a new seed.
    # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    #
    # img = tfa.image.rotate(img,
    #                        angles=tf.random.uniform(shape=[], minval=0.0, maxval=2*np.pi),
    #                        interpolation='bilinear')
    #
    # # apply series of transformations
    # img = tf.image.resize_with_crop_or_pad(img, img_height + 6, img_width + 6)
    #
    # # (i) contrast
    # img = tf.image.stateless_random_contrast(img, 0.5, 1.0, new_seed)
    #
    # # (ii) flip vertically
    # img = tf.image.stateless_random_flip_left_right(img, new_seed)
    #
    # # (iii) flip horizontally
    # img = tf.image.stateless_random_flip_up_down(img, new_seed)
    #
    # # (iv) add random hue
    # img = tf.image.stateless_random_hue(img, 0.05, new_seed)
    #
    # # (v) add noise to image
    # img = tf.image.stateless_random_jpeg_quality(img, 85, 100, new_seed)
    #
    # # (vi) random saturation
    # img = tf.image.stateless_random_saturation(img, 0.7, 1.0, new_seed)
    #
    # # (vii) Random crop back to the original size.
    # img = tf.image.stateless_random_crop(img, size=[img_height, img_width, 3], seed=new_seed)
    #
    # # (viii) Random brightness
    # img = tf.image.stateless_random_brightness(img, max_delta=0.18, seed=new_seed)
    #
    # img = tf.clip_by_value(img, 0, 1)

    return img, label


# Create a wrapper function for updating seeds.
def f(x, y):
    seed = rng.make_seeds(2)[0]
    image, label = augment((x, y), seed)
    return image, label


np.random.shuffle(list_ds)
list_ds = tf.data.Dataset.from_tensor_slices(list_ds)
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
resampled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# validation
list_ds = tf.data.Dataset.from_tensor_slices(files_val)
val_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = (
    resampled_ds
    .shuffle(1000)
    .map(f, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy', custom_kappa_metric_ohe],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=epochs)
