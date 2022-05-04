import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import sys
import random

if len(sys.argv) > 1:
    arg = sys.argv[1]
    first_run = True if (arg == '--first_run') else False
else:
    first_run = False

# define parameters for the loader
batch_size = 32
img_height = 28
img_width = 28
epochs = 10
num_classes = 10
class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
# sample_size = 128

# define this for later
AUTOTUNE = tf.data.AUTOTUNE

# move images into folders according to level
# train labels
path = os.path.join(os.getcwd(), '../datasets/retinopathy/fashion_mnist/')
train_labels = pd.read_csv(os.path.join(path, '../', 'trainLabels_mnist.csv'), dtype='string')

# remove one file which could not be processed
# train_labels = train_labels.loc[train_labels.image != '492_right']

if first_run:
    for level in class_names:
        labels = train_labels.loc[train_labels.level == level]
        os.mkdir(os.path.join(path, level))

        for image in labels['image']:
            os.rename(path + image + '.jpeg', path + level + "/" + image + '.jpeg')


def custom_kappa_metric(y_true, y_pred):
    # get sparse labels
    y_true = tf.math.reduce_sum(y_true, axis=-1, keepdims=False)

    # same for y_pred
    y_pred = tf.where(tf.less_equal(y_pred, 0.5), 0, 1)
    y_pred = tf.math.cumprod(y_pred, axis=1)
    y_pred = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)

    metric = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=True)
    metric.update_state(y_true, y_pred)
    result = metric.result()

    return result


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.validation_data = val_ds

    def on_epoch_end(self, epoch, logs={}):
        y_true = tf.concat([y for x, y in self.validation_data], axis=0)
        y_pred = tf.convert_to_tensor(self.model.predict(self.validation_data))
        kappa = custom_kappa_metric(y_true, y_pred)
        print('\nkappa is ' + kappa.numpy().astype(str))


def custom_kappa_loss(y_true, y_pred):
    # write y_true as OHE
    y_true = tf.math.reduce_sum(y_true, axis=-1, keepdims=False)
    y_true = tf.one_hot(y_true, num_classes)

    # same for y_pred
    y_pred = tf.where(tf.less_equal(y_pred, 0.5), 0, 1)
    y_pred = tf.math.cumprod(y_pred, axis=1)
    y_pred = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    y_pred = tf.one_hot(y_pred, num_classes)

    kappa_loss = tfa.losses.WeightedKappaLoss(num_classes=num_classes)
    return kappa_loss(y_true, y_pred)


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    # get the id of the image
    img_id = parts[-1]

    # determine the label
    # breakpoint()
    x = tf.strings.to_number(parts[-2], tf.int32)
    # y = tf.strings.to_number(class_names, tf.int32)

    # The second to last is the class-directory
    # ordinal = tf.math.greater_equal(x, y)

    # Integer encode the label
    return tf.cast(tf.one_hot(x, num_classes), tf.int32), img_id


def load_image(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)

    return tf.image.resize_with_pad(img, img_height, img_width)


def process_path(file_path):
    label = get_label(file_path)
    img = load_image(file_path)
    return img, label[0]


def process_path_test(file_path):
    label = get_label(file_path)
    img = load_image(file_path)
    return img, label[1]


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def predict(ds):
    y_pred = model.predict(ds)
    y_pred = np.where(np.less_equal(y_pred, 0.5), 0, 1)
    y_pred = np.cumprod(y_pred, axis=1)

    return np.sum(y_pred, axis=-1, keepdims=False)


# process images
data_dir = os.getcwd() + '/../datasets/retinopathy/fashion_mnist/'
# data_dir_test = os.getcwd() + '/../datasets/retinopathy/test_images_512/'

# (1) randomly sample 20% of the files to get the validation set
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]
files_val = random.sample(files, int(0.1*len(files)))
# files_test = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir_test + '/')) for f in fn]

# (2) the get the list of files for each class, removing the validation files
# will use the weights vector in the next step
list_ds = np.array([])
for level in class_names:
    files_train = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + level)) for f in fn]
    files_train = [elem for elem in files_train if elem not in files_val]

    # ids = np.arange(len(files_train))
    # choices = np.random.choice(ids, sample_size)
    # files_train = np.asarray(files_train)[choices]
    list_ds = np.append(list_ds, files_train)


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
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# define initializer
initializer = tf.keras.initializers.HeNormal()

# build model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(
  train_ds,
  # validation_data=val_ds,
  epochs=epochs #,callbacks=[MetricsCallback(val_ds)]
)