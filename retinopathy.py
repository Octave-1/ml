import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# define parameters for the loader
batch_size = 8
img_height = 256
img_width = 256
epochs = 1
num_classes = 5

# define this for later
AUTOTUNE = tf.data.AUTOTUNE

# move images into folders according to level
# train labels
# train_labels = pd.read_csv(os.getcwd() + '/dataset/additional/trainLabels.csv')

# for level in ['0', '1', '2', '3', '4']:
#    train_labels['level'] = train_labels['level'].astype(str)
#    labels = train_labels.loc[train_labels.level == level]
#    for image in labels['image']:
#        os.rename("dataset/train/" + image + '.jpeg', "dataset/train/" + level + "/" + image + '.jpeg')


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    x = tf.strings.to_number(parts[-2], tf.int32)
    y = tf.strings.to_number(class_names, tf.int32)

    # The second to last is the class-directory
    ordinal = tf.math.greater_equal(x, y)

    # Integer encode the label
    return tf.cast(ordinal, tf.int32)


def get_label_ohe(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names

    # Integer encode the label
    return tf.cast(one_hot, tf.int32)


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize_with_pad(img, img_height, img_width)


def process_path(file_path):
    label = get_label_ohe(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# process images
data_dir = pathlib.Path(os.getcwd() + '/dataset/train/')

# retrieve class names
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

image_count = len(list(data_dir.glob('*/*.jpeg')))
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# train & validation
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# configure dataset for performance
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# build model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])


model.compile(optimizer='sgd', loss=tfa.losses.WeightedKappaLoss(num_classes=num_classes))
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
