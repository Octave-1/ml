"""
https://www.kaggle.com/competitions/diabetic-retinopathy-detection/
using a tensorflow CNN & scikit learn to predict diabetic retinopathy
in patients, given images of both their eyes.
- training was carried out using a VastAI instance on a custom docker
container (see docker file in repo)
- the images were pre-processed & downsized (see pre_process.py)
- the pre-processed images were stored on GCP & downloaded for
model training (see setup.sh)
- this version achieves a quadratic kappa score of ~ 0.7 on the
private leaderboard
"""
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os
import sys
import random
import csv
import datetime
from sklearn import linear_model
from sklearn.metrics import cohen_kappa_score
print(tf.__version__)

if len(sys.argv) > 1:
    arg = sys.argv[1]
    first_run = True if (arg == '--first_run') else False
else:
    first_run = False


# helper function for getting image numbers
def extract_number(image_id):
    """
    The dataset contains images of both the left & right
    eyes of each patient. At test & validation time pairs
    of eye images are fed into a simple linear regression
    in order to fine-tune thresholds. This function
    extracts the image number given the image_id.
    :param image_id: e.g. 1001_right
    :return: image number '/1001_'
    """
    image_id = image_id.split('_')
    image_id = image_id[0]
    return '/' + image_id + '_'

# (0) initial setup
# set some parameters
batch_size = 64
img_height = 512
img_width = 768
epochs = 50
sample_size = 9000
val_proportion = 0.15

# setting up
num_classes = 5
class_names = np.array(['0', '1', '2', '3', '4'])
data_dir = os.getcwd() + '/../datasets/retinopathy/train_images_processed/'
data_dir_test = os.getcwd() + '/../datasets/retinopathy/test_images_processed/'

# create a generator to augment dataset
rng = tf.random.Generator.from_seed(123, alg='philox')
# this will come in handy later when reading in data using tf.Dataset
AUTOTUNE = tf.data.AUTOTUNE

# train labels
path = os.path.join(os.getcwd(), '../datasets/retinopathy/train_images_processed/')
train_labels = pd.read_csv(os.path.join(path, '../', 'trainLabels.csv'), dtype='string')

# remove one file which could not be processed
train_labels = train_labels.loc[train_labels.image != '492_right']

# if script is being run for first time, move images into folders according to level
if first_run:
    for level in class_names:
        labels = train_labels.loc[train_labels.level == level]
        os.mkdir(os.path.join(path, level))

        for image in labels['image']:
            os.rename(path + image + '.jpeg', path + level + "/" + image + '.jpeg')

# (1) extract the training files
files_train = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]

# randomly sample ~ 15% of the files to get the validation set
# ensure pairs of images are retained in validation set
images_val = random.sample(list(train_labels.image), int(val_proportion/2 * len(train_labels.image)))
images_val = [extract_number(j) for j in images_val]
images_val_no_dupes = []
[images_val_no_dupes.append(x) for x in images_val if x not in images_val_no_dupes]
files_val = [string for string in files_train if any(substring in string for substring in images_val_no_dupes)]
files_val.sort()

# get the test images
files_test = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir_test)) for f in fn]
files_test.sort()

# (2) the get the list of files for each class, removing the validation files
# since the classes are unbalanced, we upsample to achieve a balanced dataset
# this is done in the loop below
files_train_upsampled = np.array([])
for level in class_names:
    files_train_by_level = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + level)) for f in fn]
    files_train_by_level = [elem for elem in files_train_by_level if elem not in files_val]

    ids = np.arange(len(files_train_by_level))
    choices = np.random.choice(ids, sample_size)
    files_train_by_level = np.asarray(files_train_by_level)[choices]
    files_train_upsampled = np.append(files_train_upsampled, files_train_by_level)


def load_image(file_path):
    """
    :param file_path: path to image file
    :return: decoded image
    """
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)

    return img


def get_label(level, ordinal_encoding=False):
    """
    This function takes a label and returns a one hot
    encoded representation. The possibility to return
    an ordinal encoding is left open and this may be
    tried in a later version
    :param level: from '0' to '4', with '4' being most severe
    :param ordinal_encoding: boolean indicating the type of
    encoding we require
    :return:
    """
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
    """
    function which gives us the decoded image and either its
    id or label, depending on whether it's test time or not
    :param file_path: path to image
    :param ordinal_encoding: one hot encoding or ordinal encoding
    :param test: test time or not
    :return:
    """
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
    """"
    :param image: tf image
    :param label:
    :return:
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, img_height, img_width)
    image = (image / 255.0)
    return image, label


def augment(image_label, seed):
    """
    Training images are augmented by random
    (i) rotations
    (ii) flips left/right
    (iii) flips up/down
    (iv) translations
    :param image_label:
    :param seed: random seed
    :return: augmented image
    """
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


def f(x, y):
    """
    Wrapper function for updating seeds
    :param x: image
    :param y: label
    :return: augmented image, label
    """
    seed = rng.make_seeds(2)[0]
    image, label = augment((x, y), seed)
    return image, label


def eye_details(row, dataset_type):
    """
    get the image number, whether the image
    belongs to the left or the right eye,
    & also the label for train/validation
    :param row: matrix row containing image info
    :param dataset_type: train, val or test
    :return:
    """
    normalized_path = os.path.normpath(row['image'])
    path_components = normalized_path.split(os.sep)

    image_id = path_components[-1].replace('.jpeg', '')
    image_id = image_id.split('_')
    row['id'] = image_id[0]
    row['eye'] = image_id[1]

    if dataset_type in ['val', 'train']:
        row['level'] = int(path_components[-2])
    else:
        row['level'] = np.NaN

    return row


def get_dataset(y_pred, files, dataset_type):
    """
    Given the predictions from the CNN, a dataset
    can be created linking probability distributions
    from both left & right eyes
    - we use the training predictions to train a new
    linear regression model
    - we take this model & come up with labels thresholds
    using the validation dataset
    - finally we use the fine-tuned linear regression model
    to make final prediction on the test set
    :param y_pred:
    :param files:
    :param dataset_type:
    :return: dataset with left & right eye probability dists, label
    """
    df = pd.DataFrame(data=y_pred,
                      columns=["x1", "x2", "x3", "x4", "x5"])

    df_files = pd.DataFrame({'image': files})
    df = pd.concat([df_files, df], axis=1)
    df = df.apply(eye_details, axis=1, args=dataset_type).drop(['image'], axis=1)

    # split the df
    df_left = df.loc[df.eye == 'left']
    df_right = df.loc[df.eye == 'right']

    # create "id_index" to ensure that extra rows are not created when re-joining the datasets
    df_left['id_index'] = df_left.groupby(['id']).cumcount() + 1
    df_right['id_index'] = df_right.groupby(['id']).cumcount() + 1
    df = pd.merge(df_left, df_right, on=['id', 'id_index'], suffixes=('_x', '_y'))

    df_left = df.drop(['eye_y', 'level_y', 'id_index'], axis=1).rename(columns={"eye_x": "eye", "level_x": "level"})
    df_right = df.drop(['eye_x', 'level_x', 'id_index'], axis=1).rename(columns={"eye_y": "eye", "level_y": "level"})

    df = pd.concat([df_left, df_right], axis=0).reset_index()

    # oversample in order to balance the training dataset
    if dataset_type == 'train':
        max_size = df['level'].value_counts().max()
        lst = [df]
        for class_index, group in df.groupby('level'):
            lst.append(group.sample(max_size - len(group), replace=True))
            df = pd.concat(lst)

    X_data = df.sort_values(['id', 'eye'], ascending=True)
    y = X_data['level']
    X_data = X_data.drop(['level', 'id'], axis=1)
    X_data = pd.get_dummies(X_data, columns=["eye"])
    X_data = X_data.drop(['index'], axis=1)

    return X_data, y


def threshold(row, thresholds):
    """
    Gives the predicted level for a given threshold
    :param row: matrix row containing model prediction (float)
    :param thresholds: np array of threshold values
    :return: row with level prediction 'level_pred'
    """
    val = row['val']

    if val < thresholds[0]:
        row['level_pred'] = 0
    elif thresholds[0] < val < thresholds[1]:
        row['level_pred'] = 1
    elif thresholds[1] < val < thresholds[2]:
        row['level_pred'] = 2
    elif thresholds[2] < val < thresholds[3]:
        row['level_pred'] = 3
    elif val > thresholds[3]:
        row['level_pred'] = 4
    else:
        row['level_pred'] = np.NaN

    return row


def predict_with_model(ds, files, thresholds):
    """
    get model predictions using the
    regression model based on the output
    of the convnet
    :param ds: dataset
    :param files: ids of image in datasets
    :param thresholds: the new fine-tuned
    thresholds produced by the regression
    model
    :return: predictions
    """
    y_pred = model.predict(ds)

    # form the dataset & apply the regression
    X_data, _ = get_dataset(y_pred, files, dataset_type='test')

    y_pred = regr.predict(X_data)

    # use thresholds to come up w/ prediction
    y_pred = pd.DataFrame(y_pred, columns=['val']).apply(threshold, axis=1, args=thresholds)

    return y_pred.level_pred

# (1) set up datasets
# ensure dataset is shuffled & set `num_parallel_calls` so multiple images are loaded/processed in parallel.
np.random.shuffle(files_train_upsampled)
resampled_ds = tf.data.Dataset.from_tensor_slices(files_train_upsampled)
resampled_ds = resampled_ds.map(lambda x: process_path(x, ordinal_encoding=False, test=False),
                                num_parallel_calls=AUTOTUNE)

# validation
val_ds = tf.data.Dataset.from_tensor_slices(files_val)
val_ds = val_ds.map(lambda x: process_path(x, ordinal_encoding=False, test=False),
                    num_parallel_calls=AUTOTUNE)

# test
test_ds = tf.data.Dataset.from_tensor_slices(files_test)
test_ds = test_ds.map(lambda x: process_path(x, ordinal_encoding=False, test=True),
                      num_parallel_calls=AUTOTUNE)

train_ds = (
        resampled_ds
        .map(f, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
)

train_ds_resampled = (
        resampled_ds
        .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
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

# (2) define, compile & train model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(96, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(320, activation='relu'),
    tf.keras.layers.Dense(160, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy', tfa.metrics.CohenKappa(num_classes=5, sparse_labels=False)])

checkpoint_filepath = '../datasets/retinopathy/checkpoints/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=epochs,
          callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model = tf.keras.models.load_model('../datasets/retinopathy/checkpoints/')

# (3) create regression model based on training predictions
print('building regression model' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Train the model using the training sets
y_pred = model.predict(train_ds_resampled)
X_train, y_train = get_dataset(y_pred, files_train_upsampled, 'train')
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# get thresholds using the validation set
print('applying regression model to validation set' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Make predictions using the testing set
y_pred = model.predict(val_ds)
X_val, y_val = get_dataset(y_pred, files_val, 'val')
y_pred = regr.predict(X_val)

df_thresholds = pd.DataFrame(columns=["t1", "t2", "t3", "t4", "kappa"])

# find the thresholds (where to split y_pred in order to maximize agreement w/ y_val)
DF = pd.concat([y_val, pd.Series(y_pred, name='val')], axis=1)

print('thresholding' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
for i in range(0, 2000):
    thresholds = np.array([])
    for j in range(0, 4):
        # create thresholds at random
        thresholds = np.append(thresholds, random.uniform(0, 4))
        thresholds = np.sort(thresholds)

    # test the threshold & find the corresponding kappa score
    DF = DF.apply(threshold, axis=1, args=thresholds)
    kappa = cohen_kappa_score(DF.level_pred, DF.level, weights='quadratic')
    df_append = pd.DataFrame(data=np.append(thresholds, kappa).reshape(1, 5),
                             columns=["t1", "t2", "t3", "t4", "kappa"])

    df_thresholds = pd.concat([df_thresholds, df_append], axis=0)

df_thresholds = df_thresholds.sort_values('kappa', ascending=False)
print('Outputting the thresholds dataframe...')
print(df_thresholds.head)
thresholds = df_thresholds[0:1][['t1', 't2', 't3', 't4']].to_numpy()[0]
print('finished thresholding' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# (4) final predictions
print('getting final predictions' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# get predictions using the regression model
y_pred = predict_with_model(test_ds, files_test, thresholds)
img_id = list(tf.concat([y for x, y in test_ds], axis=0).numpy())
img_id = [x.decode('utf-8').rstrip('.jpeg') for x in img_id]

df_pred = pd.DataFrame({'image': img_id,
                        'level': list(y_pred)})

df_pred_man = pd.DataFrame({'image': ['25313_right', '25313_left', '27096_right', '27096_left'],
                            'level': [0, 0, 0, 0]})

df_pred = pd.concat([df_pred, df_pred_man])
df_pred['level'] = df_pred['level'].astype('int')
df_pred.to_csv('predictions_with_model.csv', header=True, index=False, quoting=csv.QUOTE_NONE)
print('done' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
