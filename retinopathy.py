import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import sys

if len(sys.argv) > 1:
    arg = sys.argv[1]
    first_run = True if (arg == '--first_run') else False
else:
    first_run = False

# define parameters for the loader
batch_size = 32
img_height = 512
img_width = 512
epochs = 2
num_classes = 5


# define this for later
AUTOTUNE = tf.data.AUTOTUNE

# move images into folders according to level
# train labels
path = os.path.join(os.getcwd(), '../datasets/retinopathy/train_images_512/')
train_labels = pd.read_csv(os.path.join(path, '../', 'trainLabels.csv'))

if first_run:
    for level in ['0', '1', '2', '3', '4']:
        train_labels['level'] = train_labels['level'].astype(str)
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
    x = tf.strings.to_number(parts[-2], tf.int32)
    y = tf.strings.to_number(class_names, tf.int32)

    # The second to last is the class-directory
    ordinal = tf.math.greater_equal(x, y)

    # Integer encode the label
    return tf.cast(ordinal[1:], tf.int32), img_id


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
data_dir = pathlib.Path(os.getcwd() + '/../datasets/retinopathy/train_images_512/')
# data_dir_test = pathlib.Path(os.getcwd() + '/../datasets/retinopathy/test_images_512/')

# retrieve class names
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

image_count = len(list(data_dir.glob('*/*.jpeg')))
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# train, validation & test
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
# test_ds = tf.data.Dataset.list_files(str(data_dir_test/'*'), shuffle=False)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# test_ds = test_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)

# configure dataset for performance
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
# test_ds = configure_for_performance(test_ds)

# build model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(96, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(160, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(192, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(num_classes - 1, activation='sigmoid')
])

# compile
model.compile(optimizer='sgd', loss='mean_squared_error', run_eagerly=True)
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[MetricsCallback(val_ds)]
)
