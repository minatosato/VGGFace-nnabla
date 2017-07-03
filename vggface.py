#! -*- coding:utf-8 -*-

from __future__ import print_function

from sklearn.model_selection import train_test_split

import numpy as np
import gzip
from tqdm import tqdm

import matplotlib.pyplot as plt

import pickle
from scipy.spatial.distance import cosine
import cv2
import time

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from nnabla.contrib.context import extension_context
ctx = extension_context("cuda.cudnn", device_id=0)
nn.set_default_context(ctx)

class VGGFace(object):
    def __init__(self, batch_size=1, load_weights=True):
        self.batch_size = batch_size
        self.x = nn.Variable([self.batch_size, 3, 224, 224])
        self.t = nn.Variable([self.batch_size, 1])
        h = F.relu(PF.convolution(self.x,  64,  (3, 3), pad=(1, 1), stride=(1, 1), name='conv1'))
        h = F.relu(PF.convolution(h,      64,  (3, 3), pad=(1, 1), stride=(1, 1), name='conv2'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h,      128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv3'))
        h = F.relu(PF.convolution(h,      128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv4'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h,      256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv5'))
        h = F.relu(PF.convolution(h,      256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv6'))
        h = F.relu(PF.convolution(h,      256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv7'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h,      512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv8'))
        h = F.relu(PF.convolution(h,      512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv9'))
        h = F.relu(PF.convolution(h,      512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv10'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h,      512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv11'))
        h = F.relu(PF.convolution(h,      512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv12'))
        h = F.relu(PF.convolution(h,      512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv13'))
        h = F.max_pooling(h, (2, 2))
        """
        flatten operation same as that of tensorflow
        """
        h = F.transpose(h, (0, 2, 3, 1)) 
        h = F.reshape(h, (batch_size, np.product(h.shape[1:])))
        """"""
        self.h = PF.affine(h, 4096, name='fc1')
        h = F.relu(self.h)
        h = PF.affine(h, 4096, name='fc2')
        h = F.relu(h)
        self.y = PF.affine(h, 2622, name='fc3')
        self.loss = F.mean(F.softmax_cross_entropy(self.y, self.t))

        if load_weights:
            self.set_pretrained_weights()

    def set_pretrained_weights(self):
        print("load pre-trained model (aroud 580MB)...")
        weights = pickle.load(open("./vggface_weights.pkl", "rb"))
        print("Done!")
        params = nn.get_parameters()
        for name in weights:
            # print(name)
            # print(params[name].d.shape)
            # print(weights[name].shape)
            assert params[name].d.shape == weights[name].shape
            params[name].d = weights[name].copy()

    def get_feature(self, images):
        batch_size = images.shape[0]
        self.__init__(batch_size=batch_size, load_weights=False)
        self.x.d = images
        self.h.forward()
        return self.h.d

def cos_smiliarity(vec1, vec2):
    return 1.0 - cosine(vec1, vec2)

def get_face(cascade, file_path):
    src_image = cv2.imread(file_path)
    src_image = cv2.cvtColor(src_image,cv2.COLOR_BGR2RGB)
    dst_image = cv2.cvtColor(src_image,cv2.COLOR_RGB2GRAY)
    facerect = cascade.detectMultiScale(dst_image, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    if len(facerect) > 0:
        for rect in facerect:
            dst_image = src_image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2],:]
        print("kenchi: " + file_path)
    else:
        print("no face")

    dst_image = cv2.resize(dst_image, (224, 224))
    # plt.imshow(dst_image)
    # plt.show()
    return dst_image

import os
#os.system("wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml")

vggface = VGGFace()
cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

start = time.time()

inp1 = get_face(cascade, "./dataset/hoge.jpg")[np.newaxis].transpose(0, 3, 1, 2)/255.


elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#inps = np.concatenate([inp1, inp2, inp3, inp4, inp5, inp6, inp7])

start = time.time()


preds = vggface.get_feature(inps)

elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

