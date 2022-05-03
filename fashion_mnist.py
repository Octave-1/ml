# TensorFlow and tf.keras
import tensorflow as tf
print(tf.__version__)
import pandas as pd
import numpy as np
import os
import random


# define parameters for the loader
batch_size = 32
img_height = 28
img_width = 28
epochs = 300
num_classes = 10
class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sample_size = 10

# define this for later
AUTOTUNE = tf.data.AUTOTUNE

# process images
data_dir = os.getcwd() + '/../datasets/retinopathy/fashion_mnist/'
# data_dir_test = os.getcwd() + '/../datasets/retinopathy/test_images_512/'

# (1) randomly sample 20% of the files to get the validation set
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]
# files_val = random.sample(files, int(0.1*len(files)))
# files_test = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir_test + '/')) for f in fn]

# (2) the get the list of files for each class, removing the validation files
# will use the weights vector in the next step
list_ds = np.array([])
for level in class_names:
    files_train = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + level)) for f in fn]
    # files_train = [elem for elem in files_train if elem not in files_val]

    ids = np.arange(len(files_train))
    choices = np.random.choice(ids, sample_size)
    files_train = np.asarray(files_train)[choices]
    list_ds = np.append(list_ds, files_train)


def load_image(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=1)

    return tf.image.resize_with_pad(img, img_height, img_width)


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


def process_path(file_path):
    label = get_label(file_path)
    img = load_image(file_path)
    return img, label[0]


np.random.shuffle(list_ds)
list_ds = tf.data.Dataset.from_tensor_slices(list_ds)
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
resampled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# validation
# list_ds = tf.data.Dataset.from_tensor_slices(files_val)
# val_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = (
    resampled_ds
    .shuffle(1000)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# val_ds = (
#    val_ds
#    .batch(batch_size)
#    .prefetch(AUTOTUNE)
# )


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Flatten(input_shape=(img_width, img_height)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, epochs=epochs)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)


# get list of train files
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

df = pd.read_csv("mnist_files.csv", names=["image", "level"])
df['level'] = train_labels
df['image'] = df['image'].str.replace('.jpeg', '', regex=False)
df.to_csv('trainLabels_mnist.csv', index=False)
