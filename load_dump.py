
keys = ['conv1/conv/W',
 'conv1/conv/b',
 'conv2/conv/W',
 'conv2/conv/b',
 'conv3/conv/W',
 'conv3/conv/b',
 'conv4/conv/W',
 'conv4/conv/b',
 'conv5/conv/W',
 'conv5/conv/b',
 'conv6/conv/W',
 'conv6/conv/b',
 'conv7/conv/W',
 'conv7/conv/b',
 'conv8/conv/W',
 'conv8/conv/b',
 'conv9/conv/W',
 'conv9/conv/b',
 'conv10/conv/W',
 'conv10/conv/b',
 'conv11/conv/W',
 'conv11/conv/b',
 'conv12/conv/W',
 'conv12/conv/b',
 'conv13/conv/W',
 'conv13/conv/b',
 'fc1/affine/W',
 'fc1/affine/b',
 'fc2/affine/W',
 'fc2/affine/b',
 'fc3/affine/W',
 'fc3/affine/b']


from collections import OrderedDict
weights = OrderedDict()
from keras_vggface.vggface import VGGFace
model = VGGFace()

model.summary()
import numpy as np

i = 0
j = 0
while(len(keys) != i):
    name = model.layers[j].name
    if (not ("fc" in name)) and (not ("conv" in name)):
        j+=1
        continue
    if "relu" in name:
        j+=1
        continue

    print(name)
    if "fc" in name:
        weights[keys[i]] = model.layers[j].get_weights()[0]
        print("    "+keys[i])
        i+=1
        weights[keys[i]] = model.layers[j].get_weights()[1]
        print("    "+keys[i])
        print("params: " + str(np.product(model.layers[j].get_weights()[0].shape) + np.product(model.layers[j].get_weights()[1].shape)))
        i+=1
    elif "conv" in name:
        weights[keys[i]] = model.layers[j].get_weights()[0].transpose(3, 2, 0, 1)
        print("    "+keys[i])
        i+=1
        weights[keys[i]] = model.layers[j].get_weights()[1]
        print("    "+keys[i])
        print("params: " + str(np.product(model.layers[j].get_weights()[0].shape) + np.product(model.layers[j].get_weights()[1].shape)))
        i+=1
    j+=1

import pickle    
pickle.dump(weights, open("vggface_weights.pkl", "wb"), protocol=2)
