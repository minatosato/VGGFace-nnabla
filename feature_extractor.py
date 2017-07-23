#! -*- coding:utf-8 -*-

from __future__ import print_function

from glob import glob
from itertools import combinations
from itertools import permutations
import random
import time

import cv2
import matplotlib.pyplot as plt
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from tqdm import tqdm
from vggface import VGGFace


from nnabla.contrib.context import extension_context
ctx = extension_context("cuda.cudnn", device_id=0)
nn.set_default_context(ctx)

def cos_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))

def get_vgg_fc_feature(model, images):
    batch_size = images.shape[0]
    if batch_size != model.batch_size:
        x = nn.Variable([batch_size, 3, 224, 224])
        model.__init__(x, include_top=True, load_weights=False)
    model.x.d = images
    model.layers['fc1'].forward()
    return model.layers['fc1'].d.copy()


x = nn.Variable([50, 3, 224, 224])
vggface = VGGFace(x, include_top=True, load_weights=True, pooling='avg')

print("loading images...")
file_name_list = []
input_images = []
for file_path in tqdm(glob("./lfw-deepfunneled/*/*.jpg")):
    file_name_list.append(file_path)
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)/255.
    input_images.append(img)
input_images = np.array(input_images)
print("done!")

batch_size = 10
n_batch = int(len(input_images) / batch_size) + 1

vgg_fc_features = []
print("getting pre-trained feature from images...")
for i in tqdm(range(n_batch)):
    batch = input_images[i*batch_size:(i+1)*batch_size]
    pred = get_vgg_fc_feature(vggface, batch)
    vgg_fc_features.append(pred)
print("done!")
vgg_fc_features = np.concatenate(vgg_fc_features)


all_pattern = True
triplets = []
print("making triplets which are lists of (anchor, positive, negative)...")
for person in tqdm(glob("./lfw-deepfunneled/*/")):
    person_files = glob(person + "*.jpg")
    if len(person_files) <= 1:
        continue
    person_indices = list(map(file_name_list.index, person_files))
    other_indices = list(set(list(range(len(file_name_list))))-set(person_indices))
    if all_pattern:
        sampling = permutations
    else:
        sampling = combinations
    pattern = sampling(person_indices, 2)
    for (anchor, pos) in pattern:
        if True:
            neg = random.choice(other_indices)
            assert anchor != neg
            assert pos != neg
            triplets.append((anchor, pos, neg))
        else:
            for neg in other_indices:
                assert anchor != neg
                assert pos != neg
                triplets.append((anchor, pos, neg))
print("done!")
triplets = np.array(triplets)



with nn.parameter_scope('face_embedding'):
    nn.clear_parameters()

def triplet_loss(phi1, phi2, phi3, alpha=0.2):
    zero = nn.Variable([batch_size, ], need_grad=False)
    zero.d = 0.
    phi_norm1 = F.pow_scalar(F.sum(phi1*phi1, axis=1, keepdims=True), 1./2.)
    phi_norm2 = F.pow_scalar(F.sum(phi2*phi2, axis=1, keepdims=True), 1./2.)
    phi_norm3 = F.pow_scalar(F.sum(phi3*phi3, axis=1, keepdims=True), 1./2.)
    with nn.parameter_scope('face_embedding'):
        x1 = PF.affine(phi1/phi_norm1, 1024)
    with nn.parameter_scope('face_embedding'):
        x2 = PF.affine(phi2/phi_norm2, 1024)
    with nn.parameter_scope('face_embedding'):
        x3 = PF.affine(phi3/phi_norm3, 1024)
    pos_norm = F.pow_scalar(F.sum((x1 - x2)**2, axis=1), 1)
    neg_norm = F.pow_scalar(F.sum((x1 - x3)**2, axis=1), 1)
    loss = F.mean(F.max(F.concatenate(zero, - neg_norm + alpha + pos_norm)))

    with nn.parameter_scope('face_embedding'):
        nn.get_parameters()['affine/b'].d = 0.0
        nn.get_parameters()['affine/b'].need_grad = False

    return loss

def face_embedding(input_vgg_fc_features):
    batch_size = input_vgg_fc_features.shape[0]
    phi = nn.Variable([batch_size, 4096])
    phi.d = input_vgg_fc_features
    with nn.parameter_scope('face_embedding'):
        x = PF.affine(phi, 1024)
    x.forward()
    return x.d.copy()

batch_size = 128
n_batch = int(len(triplets)/batch_size)
alpha = 0.2
phi1 = nn.Variable([batch_size, 4096]) # anchor
phi2 = nn.Variable([batch_size, 4096]) # positive
phi3 = nn.Variable([batch_size, 4096]) # negative

loss = triplet_loss(phi1, phi2, phi3, alpha=alpha)
# embedding = face_embedding(phi1)


solver = S.Sgd(0.01)
with nn.parameter_scope('face_embedding'):
    solver.set_parameters(nn.get_parameters())

use_pre_trained_face_embeddings = True
if use_pre_trained_face_embeddings:
    with nn.parameter_scope('face_embedding'):
        nn.load_parameters("./face_embedding.h5")
else:
    for epoch in range(10):
        print("epoch " + str(epoch) + ": ", end="")
        sum_loss = 0.0
        triplets = triplets[np.random.permutation(len(triplets))]
        for i in tqdm(range(n_batch)):
            tmp = triplets[i*batch_size:(i+1)*batch_size]
            a = tmp[:,0]
            p = tmp[:,1]
            n = tmp[:,2]
            phi1.d = vgg_fc_features[a]
            phi2.d = vgg_fc_features[p]
            phi3.d = vgg_fc_features[n]
            loss.forward()
            solver.zero_grad()
            loss.backward()
            # solver.weight_decay(1e-3)
            solver.update()
            sum_loss += loss.d.copy()
        print("loss: " + str(sum_loss/n_batch))
    with nn.parameter_scope('face_embedding'):
        nn.save_parameters('./face_embedding.h5')


test_file_name = []
test_input_images = []
for file_path in tqdm(glob("./dataset/*/*.jpg")):
    test_file_name.append(file_path)
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis].transpose(0, 3, 1, 2)/255.
    test_input_images.append(img)
test_input_images = np.concatenate(test_input_images)
test_preds = get_vgg_fc_feature(vggface, test_input_images)
embeddings = face_embedding(test_preds)
# embeddings = test_preds

def get_image_feature(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis].transpose(0, 3, 1, 2)/255.
    pred = get_vgg_fc_feature(vggface, img)[0]
    return face_embedding(pred)[0]
    # return pred

query = get_image_feature("./test/query_jobs.jpg")

print("".ljust(35) + "L2norm   cos similarity")
for i,e in enumerate(embeddings):
    print(test_file_name[i].ljust(33) + ": ", end="")
    print(np.sqrt(((query - e)**2).sum()), end="  ")
    print(cos_similarity(query, e))