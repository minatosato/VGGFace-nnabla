#! -*- coding:utf-8 -*-

from __future__ import print_function
# from __future__ import absolute_import

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
    def __init__(self, include_top=True, batch_size=50, load_weights=True):
        self.batch_size = batch_size
        self.x = nn.Variable([self.batch_size, 3, 224, 224])
        self.t = nn.Variable([self.batch_size, 1])
        h = F.relu(PF.convolution(self.x, 64, (3, 3), pad=(1, 1), stride=(1, 1), name='conv1'))
        h = F.relu(PF.convolution(h, 64,  (3, 3), pad=(1, 1), stride=(1, 1), name='conv2'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h, 128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv3'))
        h = F.relu(PF.convolution(h, 128, (3, 3), pad=(1, 1), stride=(1, 1), name='conv4'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv5'))
        h = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv6'))
        h = F.relu(PF.convolution(h, 256, (3, 3), pad=(1, 1), stride=(1, 1), name='conv7'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv8'))
        h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv9'))
        h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv10'))
        h = F.max_pooling(h, (2, 2))
        h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv11'))
        h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv12'))
        h = F.relu(PF.convolution(h, 512, (3, 3), pad=(1, 1), stride=(1, 1), name='conv13'))
        h = F.max_pooling(h, (2, 2))
        if include_top:
            """
            flatten operation same as that of tensorflow
            """
            h = F.transpose(h, (0, 2, 3, 1)) 
            h = F.reshape(h, (batch_size, np.product(h.shape[1:])))
            """"""
            h = PF.affine(h, 4096, name='fc1')
            h = F.relu(h)
            h = PF.affine(h, 4096, name='fc2')
            h = F.relu(h)
            h = PF.affine(h, 2622, name='fc3')
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

