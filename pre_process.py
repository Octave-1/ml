import cv2
import glob
import os
import numpy
from datetime import datetime


def scale_radius(img, scale):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


scale = 500
c = 0
images = glob.glob("../datasets/retinopathy/test_images/*.jpeg")

for f in images:
    try:
        a = cv2.imread(f)
        # scale img to a given radius
        a = scale_radius(a, scale)
        # subtract local mean color
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
        # remove outer 10%
        b = numpy.zeros(a.shape)
        cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        a = a * b + 128 * (1 - b)
        file_name = os.path.split(f)[1]
        path = '../datasets/retinopathy/test_images_processed/' + file_name
        cv2.imwrite(path, a)
        c = c + 1
        if c % 1000 == 0:
            print(str(c) + ' images processed...')
            now = datetime.now()
            print('time:', now)

    except:
        print(f)
