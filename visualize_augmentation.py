import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# examine transformation effects on full-size images
def visualize(original, augmented, label):
    # define subplots
    fig, ax = plt.subplots(1, 2, figsize=(40, 14))
    fig.tight_layout()

    # create subplots
    ax[0].imshow(original)
    ax[0].set_title('Original image: ' + str(label))
    ax[1].imshow(augmented)
    ax[1].set_title('Augmented image')
    plt.show()


# create a generator to augment dataset
rng = tf.random.Generator.from_seed(123, alg='philox')
new_seed = tf.random.experimental.stateless_split(rng.make_seeds(2)[0], num=1)[0, :]
image = Image.open('../datasets/retinopathy/train_images_processed/4/16_right.jpeg')
image = np.array(image)
# augmented = f(image, 4)[0]
img = tfa.image.rotate(image,
                       angles=tf.random.uniform(shape=[], minval=0.0, maxval=2*np.pi),
                       interpolation='bilinear',
                       fill_mode='constant',
                       fill_value=np.mean(image))

img_height = 512
img_width = 768

# apply series of transformations
img = tf.image.resize_with_crop_or_pad(image, img_height + 6, img_width + 6)

# (vii) Random crop back to the original size.
img = tf.image.stateless_random_crop(image, size=[img_height, img_width, 3], seed=new_seed)

# (i) contrast
img = tf.image.stateless_random_contrast(image, 0.5, 1.0, new_seed)

# (ii) flip vertically
img = tf.image.stateless_random_flip_left_right(image, new_seed)

# (iii) flip horizontally
img = tf.image.stateless_random_flip_up_down(image, new_seed)

# (iv) add random hue
img = tf.image.stateless_random_hue(image, 0.05, new_seed)

# (v) add noise to image
img = tf.image.stateless_random_jpeg_quality(image, 85, 100, new_seed)
# (vi) random saturation
img = tf.image.stateless_random_saturation(image, 0.7, 1.0, new_seed)

# (viii) Random brightness
img = tf.image.stateless_random_brightness(image, max_delta=0.18, seed=new_seed)

visualize(image, img, 4)

