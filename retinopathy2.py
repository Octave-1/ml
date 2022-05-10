# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os
import sys
import random
import csv

print(tf.__version__)

# define this for later
AUTOTUNE = tf.data.AUTOTUNE

if len(sys.argv) > 1:
    arg = sys.argv[1]
    first_run = True if (arg == '--first_run') else False
else:
    first_run = False

# ----------------- define parameters ----------------- #
batch_size = 32
img_height = 512
img_width = 768
epochs = 50
sample_size = 9000
val_proportion = 0.15
# determine if we use the ordinal target vector or the one hot encoding version
ordinal_encoding = False
# --------------- end define parameters --------------- #

num_classes = 5
class_names = np.array(['0', '1', '2', '3', '4'])
data_dir = os.getcwd() + '/../datasets/retinopathy/train_images_processed/'
data_dir_test = os.getcwd() + '/../datasets/retinopathy/test_images_processed/'

# create a generator to augment dataset
rng = tf.random.Generator.from_seed(123, alg='philox')

# train labels
path = os.path.join(os.getcwd(), '../datasets/retinopathy/train_images_processed/')
train_labels = pd.read_csv(os.path.join(path, '../', 'trainLabels.csv'), dtype='string')

# remove one file which could not be processed
train_labels = train_labels.loc[train_labels.image != '492_right']

# move images into folders according to level
if first_run:
    for level in class_names:
        labels = train_labels.loc[train_labels.level == level]
        os.mkdir(os.path.join(path, level))

        for image in labels['image']:
            os.rename(path + image + '.jpeg', path + level + "/" + image + '.jpeg')

# (1) randomly sample 10% of the files to get the validation set
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]
files_val = random.sample(files, int(val_proportion * len(files)))
files_test = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir_test)) for f in fn]

# (2) the get the list of files for each class, removing the validation files
list_ds = np.array([])
for level in class_names:
    files_train = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + level)) for f in fn]
    files_train = [elem for elem in files_train if elem not in files_val]

    ids = np.arange(len(files_train))
    choices = np.random.choice(ids, sample_size)
    files_train = np.asarray(files_train)[choices]
    list_ds = np.append(list_ds, files_train)


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


def get_label(level, ordinal_encoding=False):
    # determine the label
    ohe = tf.strings.to_number(level, tf.int32)
    classes = tf.strings.to_number(class_names, tf.int32)

    if ordinal_encoding:
        # The second to last is the class-directory
        output = tf.math.greater_equal(ohe, classes)[1:]
    else:
        output = tf.one_hot(ohe, num_classes)

    # Integer encode the label
    return tf.cast(output, tf.int32)


def process_path(file_path, ordinal_encoding=False, test=False):
    # get the id of the image
    parts = tf.strings.split(file_path, os.path.sep)
    img_id = parts[-1]
    img = load_image(file_path)

    if not test:
        label = get_label(parts[-2], ordinal_encoding)
        output = (img, label)
    else:
        output = (img, img_id)

    return output


def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, img_height, img_width)
    image = (image / 255.0)
    return image, label


def augment(image_label, seed):
    img, label = image_label
    img, label = resize_and_rescale(img, label)

    # Make a new seed.
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    # (i) rotate randomly
    img_mean = tf.math.reduce_mean(img)
    img = tfa.image.rotate(img,
                           angles=tf.random.uniform(shape=[], minval=0.0, maxval=2*np.pi),
                           interpolation='bilinear',
                           fill_mode='constant',
                           fill_value=img_mean)

    # (ii) flip vertically
    img = tf.image.stateless_random_flip_left_right(img, new_seed)

    # (iii) flip horizontally
    img = tf.image.stateless_random_flip_up_down(img, new_seed)

    # (iv)
    img = tfa.image.translate(images=img,
                              translations=[tf.random.uniform(shape=[], minval=-0.05*img_width, maxval=0.05*img_width),
                                            tf.random.uniform(shape=[], minval=-0.05*img_height, maxval=0.05*img_height)],
                              interpolation='nearest',
                              fill_mode='constant',
                              fill_value=img_mean)

    img = tf.clip_by_value(img, 0, 1)

    return img, label


# Create a wrapper function for updating seeds.
def f(x, y):
    seed = rng.make_seeds(2)[0]
    image, label = augment((x, y), seed)
    return image, label


def predict(ds):
    y_pred = model.predict(ds)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred


# adjust code based on ordinal vs OHE configuration
if ordinal_encoding:
    activation = 'sigmoid'
    metric = custom_kappa_metric
    loss = tf.keras.losses.MeanSquaredError()
    num_nodes = num_classes - 1
else:
    activation = 'softmax'
    metric = custom_kappa_metric_ohe
    loss = tf.keras.losses.CategoricalCrossentropy()
    num_nodes = num_classes

# ensure dataset is shuffled & set `num_parallel_calls` so multiple images are loaded/processed in parallel.
np.random.shuffle(list_ds)
list_ds = tf.data.Dataset.from_tensor_slices(list_ds)
resampled_ds = list_ds.map(lambda x: process_path(x, ordinal_encoding=ordinal_encoding, test=False),
                           num_parallel_calls=AUTOTUNE)

# validation
list_ds = tf.data.Dataset.from_tensor_slices(files_val)
val_ds = list_ds.map(lambda x: process_path(x, ordinal_encoding=ordinal_encoding, test=False),
                     num_parallel_calls=AUTOTUNE)

# test
list_ds = tf.data.Dataset.from_tensor_slices(files_test)
test_ds = list_ds.map(lambda x: process_path(x, ordinal_encoding=ordinal_encoding, test=True),
                      num_parallel_calls=AUTOTUNE)

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

test_ds = (
        test_ds
        .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(96, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_nodes, activation=activation)
])

model.compile(optimizer='adam',
              loss=loss,
              metrics=['accuracy'],
              run_eagerly=True)

checkpoint_filepath = '../datasets/retinopathy/checkpoints/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=epochs,
          callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

# get predictions
y_pred = predict(test_ds)
img_id = list(tf.concat([y for x, y in test_ds], axis=0).numpy())
img_id = [x.decode('utf-8').rstrip('.jpeg') for x in img_id]

df_pred = pd.DataFrame({'image': img_id,
                        'level': list(y_pred)})

df_pred_man = pd.DataFrame({'image': ['25313_right', '27096_right'],
                            'level': [0, 0]})

df_pred = pd.concat([df_pred, df_pred_man])

df_pred.to_csv('predictions.csv', header=True, index=False, quoting=csv.QUOTE_NONE)
