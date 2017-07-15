#! -*- coding:utf-8 -*-

from __future__ import print_function

from vggface import VGGFace

import numpy as np

import matplotlib.pyplot as plt

import cv2
import time

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from nnabla.contrib.context import extension_context
ctx = extension_context("cuda.cudnn", device_id=0)
nn.set_default_context(ctx)

def cos_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))

def get_feature(model, images):
    batch_size = images.shape[0]
    if batch_size != model.batch_size:
        model.__init__([batch_size, 3, 224, 224], include_top=True, load_weights=False)
    model.x.d = images
    model.output.forward()
    return model.output.d.copy()

def get_face(cascade_classifier, file_path):
    src_image = cv2.imread(file_path)
    src_image = cv2.cvtColor(src_image,cv2.COLOR_BGR2RGB)
    dst_image = cv2.cvtColor(src_image,cv2.COLOR_RGB2GRAY)
    facerect = cascade_classifier.detectMultiScale(dst_image, scaleFactor=1.1, minNeighbors=3, minSize=(1, 1))

    if len(facerect) > 0:
        # for rect in facerect:
        # img_size_x = dst_image.shape[0]
        # img_size_y = dst_image.shape[1]

        rect = facerect[-1]
        width = rect[2]
        height = rect[3]
        x = rect[0]
        y = rect[1]
        # x = x - int(width*0.4)
        # if x<0:
        #     x = 0
        # y = y - int(height*0.4)
        # if y<0:
        #     y = 0
        # width = int(rect[2]*1.8)
        # height = int(rect[3]*1.8)
        # dst = image[y:y+height, x:x+width]

        # print(rect)
        dst_image = src_image[y:y+height, x:x+width,:]
        # plt.imshow(dst_image)
        # cv2.imwrite(file_path + "_.png", dst_image)
        # plt.show()
        print("kenchi: " + file_path)
    else:
        print("no face")

    dst_image = cv2.resize(dst_image, (224, 224))
    # plt.imshow(dst_image)
    # plt.show()
    return dst_image

x = nn.Variable([50, 3, 224, 224])
vggface = VGGFace(x, include_top=True, load_weights=True)

# cascade_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
cascade_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")



start = time.time()
from glob import glob

inputs = []
for file in glob("./dataset/*.jpg"):
    inputs.append(get_face(cascade_classifier, file)[np.newaxis].transpose(0, 3, 1, 2)/255.)


elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

inputs = np.concatenate(inputs)


start = time.time()


preds = get_feature(vggface, inputs)

elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

