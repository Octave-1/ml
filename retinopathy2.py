# TensorFlow and tf.keras
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

# define this for later
AUTOTUNE = tf.data.AUTOTUNE


# helper function for getting image numbers
def extract_number(image_id):
    image_id = image_id.split('_')
    image_id = image_id[0]
    return '/' + image_id + '_'

if len(sys.argv) > 1:
    arg = sys.argv[1]
    first_run = True if (arg == '--first_run') else False
else:
    first_run = False

# ----------------- define parameters ----------------- #
batch_size = 64
img_height = 512
img_width = 768
epochs = 10
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
files_train = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]

# ensure pairs of images are retained in validation set
images_val = random.sample(list(train_labels.image), int(val_proportion/2 * len(train_labels.image)))
images_val = [extract_number(j) for j in images_val]
images_val_no_dupes = []
[images_val_no_dupes.append(x) for x in images_val if x not in images_val_no_dupes]
files_val = [string for string in files_train if any(substring in string for substring in images_val_no_dupes)]
files_val.sort()

# test
files_test = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir_test)) for f in fn]
files_test.sort()

# (2) the get the list of files for each class, removing the validation files
files_train_upsampled = np.array([])
for level in class_names:
    files_train_by_level = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + level)) for f in fn]
    files_train_by_level = [elem for elem in files_train_by_level if elem not in files_val]

    ids = np.arange(len(files_train_by_level))
    choices = np.random.choice(ids, sample_size)
    files_train_by_level = np.asarray(files_train_by_level)[choices]
    files_train_upsampled = np.append(files_train_upsampled, files_train_by_level)


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


def predict_with_model(ds, files, thresholds):
    """get model predictions using the
       regression model as well as the
       output of the convnet
    """
    y_pred = model.predict(ds); data = pd.DataFrame(data=y_pred); data.to_csv('y_pred3.csv', header=False, index=False, quoting=csv.QUOTE_NONE)

    # form the dataset & apply the regression
    X_data, _ = get_dataset(y_pred, files, dataset_type='test'); data = pd.DataFrame(data=files); data.to_csv('files_test.csv', header=False, index=False, quoting=csv.QUOTE_NONE)

    y_pred = regr.predict(X_data)

    # use thresholds to come up w/ prediction
    y_pred = pd.DataFrame(y_pred, columns=['val']).apply(threshold, axis=1, args=[thresholds])

    return y_pred.level_pred


def threshold(row, thresholds):
    val = row['val']

    if val < thresholds[0]:
        row['level_pred'] = 0
    elif thresholds[0] < val < thresholds[1]:
        row['level_pred'] = 1
    elif thresholds[1] < val < thresholds[2]:
        row['level_pred'] = 2
    elif thresholds[2] < val < thresholds[3]:
        row['level_pred'] = 3
    else:
        row['level_pred'] = 4

    return row


def eye_details(row, dataset_type):

    normalized_path = os.path.normpath(row['image'])
    path_components = normalized_path.split(os.sep)

    # get the image number & also whether the image
    # belongs to the left or the right eye
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

    df = pd.DataFrame(data=y_pred,
                      columns=["x1", "x2", "x3", "x4", "x5"])

    df_files = pd.DataFrame({'image': files})
    df = pd.concat([df_files, df], axis=1)
    df = df.apply(eye_details, axis=1, args=[dataset_type]).drop(['image'], axis=1)

    # split the df
    df_left = df.loc[df.eye == 'left']
    df_right = df.loc[df.eye == 'right']

    df_left['id_index'] = df_left.groupby(['id']).cumcount() + 1
    df_right['id_index'] = df_right.groupby(['id']).cumcount() + 1
    df = pd.merge(df_left, df_right, on=['id', 'id_index'], suffixes=('_x', '_y'))

    df_left = df.drop(['eye_y', 'level_y', 'id_index', 'id'], axis=1).rename(columns={"eye_x": "eye", "level_x": "level"})
    df_right = df.drop(['eye_x', 'level_x', 'id_index', 'id'], axis=1).rename(columns={"eye_y": "eye", "level_y": "level"})

    df = pd.concat([df_left, df_right], axis=0).reset_index()

    if dataset_type == 'train':
        max_size = df['level'].value_counts().max()
        lst = [df]
        for class_index, group in df.groupby('level'):
            lst.append(group.sample(max_size - len(group), replace=True))
            df = pd.concat(lst)

    y = df['level']
    X_data = df.drop(['level'], axis=1)

    X_data = pd.get_dummies(X_data, columns=["eye"])
    X_data = X_data.drop(['index'], axis=1)

    return X_data, y


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
np.random.shuffle(files_train_upsampled)
resampled_ds = tf.data.Dataset.from_tensor_slices(files_train_upsampled)
resampled_ds = resampled_ds.map(lambda x: process_path(x, ordinal_encoding=ordinal_encoding, test=False),
                                num_parallel_calls=AUTOTUNE)

# validation
val_ds = tf.data.Dataset.from_tensor_slices(files_val)
val_ds = val_ds.map(lambda x: process_path(x, ordinal_encoding=ordinal_encoding, test=False),
                    num_parallel_calls=AUTOTUNE)

# test
test_ds = tf.data.Dataset.from_tensor_slices(files_test)
test_ds = test_ds.map(lambda x: process_path(x, ordinal_encoding=ordinal_encoding, test=True),
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
    tf.keras.layers.Dense(num_nodes, activation=activation)
])

model.compile(optimizer='adam',
              loss=loss,
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

print('building regression model' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Train the model using the training sets
y_pred = model.predict(train_ds_resampled); data = pd.DataFrame(data=y_pred); data.to_csv('y_pred1.csv', header=False, index=False, quoting=csv.QUOTE_NONE)
X_train, y_train = get_dataset(y_pred, files_train_upsampled, 'train'); data = pd.DataFrame(data=files_train_upsampled); data.to_csv('files_train_upsampled.csv', header=False, index=False, quoting=csv.QUOTE_NONE)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print('applying regression model to validation set' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Make predictions using the testing set
y_pred = model.predict(val_ds); data = pd.DataFrame(data=y_pred); data.to_csv('y_pred2.csv', header=False, index=False, quoting=csv.QUOTE_NONE)
X_val, y_val = get_dataset(y_pred, files_val, 'val'); data = pd.DataFrame(data=files_val); data.to_csv('files_val.csv', header=False, index=False, quoting=csv.QUOTE_NONE)
y_pred = regr.predict(X_val)

df_thresholds = pd.DataFrame(columns=["t1", "t2", "t3", "t4", "kappa"])

# find the thresholds (where to split y_pred in order to maximize agreement w/ y_val)
DF = pd.concat([y_val, pd.Series(y_pred, name='val')], axis=1)

print('thresholding' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
for i in range(0, 2000):
    thresholds = np.array([])
    for j in range(0, 4):
        thresholds = np.append(thresholds, random.uniform(0, 4))
        thresholds = np.sort(thresholds)

    DF = DF.apply(threshold, axis=1, args=[thresholds])
    kappa = cohen_kappa_score(DF.level_pred, DF.level, weights='quadratic')
    df_append = pd.DataFrame(data=np.append(thresholds, kappa).reshape(1, 5),
                             columns=["t1", "t2", "t3", "t4", "kappa"])

    df_thresholds = pd.concat([df_thresholds, df_append], axis=0)

df_thresholds = df_thresholds.sort_values('kappa', ascending=False)
print('Outputting the thresholds dataframe...')
print(df_thresholds.head)
thresholds = df_thresholds[0:1][['t1', 't2', 't3', 't4']].to_numpy()[0]
print('finished thresholding' + ' @ ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# get predictions
y_pred = predict(test_ds)
img_id = list(tf.concat([y for x, y in test_ds], axis=0).numpy())
img_id = [x.decode('utf-8').rstrip('.jpeg') for x in img_id]

df_pred = pd.DataFrame({'image': img_id,
                        'level': list(y_pred)})

df_pred_man = pd.DataFrame({'image': ['25313_right', '25313_left', '27096_right', '27096_left'],
                            'level': [0, 0, 0, 0]})

df_pred = pd.concat([df_pred, df_pred_man])

print('writing predictions...')
df_pred.to_csv('predictions.csv', header=True, index=False, quoting=csv.QUOTE_NONE)

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