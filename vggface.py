#! -*- coding:utf-8 -*-

from __future__ import print_function
# from __future__ import absolute_import

from collections import OrderedDict
import pickle
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S


"""cuda setting"""
from nnabla.contrib.context import extension_context
ctx = extension_context("cuda.cudnn", device_id=0)
nn.set_default_context(ctx)
""""""

class VGGFace(object):
    def __init__(self, input_shape, include_top=True, load_weights=True):
        self.batch_size = input_shape[0]
        self.x = nn.Variable(input_shape)
        self.t = nn.Variable([self.batch_size, 1])
        self.layers = OrderedDict()
        h = self.layers['conv1'] = F.relu(PF.convolution(self.x, 64, (3, 3), pad=(1, 1), stride=(1, 1), name='conv1'))
        h = self.layers['conv2'] = F.relu(PF.convolution(h, 64,  (3, 3), pad=(1, 1), stride=(1, 1), name='conv2'))
        h = self.layers['pool1'] = F.max_pooling(h, (2, 2))
        h = self.layers['conv3'] = F.relu(PF.convolution(h, 128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv3'))
        h = self.layers['conv4'] = F.relu(PF.convolution(h, 128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv4'))
        h = self.layers['pool2'] = F.max_pooling(h, (2, 2))
        h = self.layers['conv5'] = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv5'))
        h = self.layers['conv6'] = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv6'))
        h = self.layers['conv7'] = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv7'))
        h = self.layers['pool3'] = F.max_pooling(h, (2, 2))
        h = self.layers['conv8'] = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv8'))
        h = self.layers['conv9'] = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv9'))
        h = self.layers['conv10'] = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv10'))
        h = self.layers['pool4'] = F.max_pooling(h, (2, 2))
        h = self.layers['conv11'] = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv11'))
        h = self.layers['conv12'] = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv12'))
        h = self.layers['conv13'] = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv13'))
        h = self.layers['pool5'] = F.max_pooling(h, (2, 2))
        if include_top:
            """
            flatten operation same as that of tensorflow
            """
            h = F.transpose(h, (0, 2, 3, 1)) 
            h = F.reshape(h, (self.batch_size, np.product(h.shape[1:])))
            """"""
            h = self.layers['fc1'] = PF.affine(h, 4096, name='fc1')
            h = self.layers['relu1'] = F.relu(h)
            h = self.layers['fc2'] = PF.affine(h, 4096, name='fc2')
            h = self.layers['relu2'] = F.relu(h)
            h = self.layers['fc3'] = PF.affine(h, 2622, name='fc3')
            # self.loss = F.mean(F.softmax_cross_entropy(self.y, self.t))

        self.output = h

        if load_weights:
            self.set_pretrained_weights(include_top)

    def set_pretrained_weights(self, include_top):
        print("load pre-trained model...")
        weights = pickle.load(open("./vggface_weights.pkl", "rb"))
        params = nn.get_parameters()
        for name in weights:
            if "fc" in name and include_top==False:
                break
            # print(name)
            assert params[name].d.shape == weights[name].shape
            params[name].d = weights[name].copy()
        print("done!")

