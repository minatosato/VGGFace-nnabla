# VGGFace-nnabla

## About
This repository is implementation of VGG-Face CNN with Sony's Neural Network Libraries, [NNabla](https://github.com/sony/nnabla).

## Requirement
- Python2
- NNabla 0.9.1rc3
- OpenCV 2.4.11

## Usage
```
$ wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
$ wget https://github.com/hogefugabar/VGGFace-nnabla/releases/download/v0.0.1-alpha/vggface_weights.pkl
```

### Feature extraction
```py
from vggface import VGGFace
import numpy as np

batch_size = 50

"""convolution feature"""
model = VGGFace([batch_size, 3, 224, 224], include_top=False)
x = model.x
feature = model.output
""""""

"""first FC layer feature"""
model = VGGFace([batch_size, 3, 224, 224], include_top=True)
x = model.x
feature = model.layers['fc1']
""""""

image = cv2.imread("./image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))[np.newaxis].transpose(0, 3, 1, 2)/255.

x.d = image
feature.forward()
print(feature.d.copy()[0]) # this is feature of ./image.jpg
```


### Finetuning
```py
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from vggface import VGGFace

n_hidden = 4096
n_classes = 10
batch_size = 50

model = VGGFace([batch_size, 3, 224, 224], include_top=False)
x = model.x
t = model.t
h = model.output
h = F.relu(PF.affine(h, n_hidden, name='fc1'))
h = F.relu(PF.affine(h, n_hidden, name='fc2'))
y = PF.affine(h, n_classes)
loss = F.mean(F.softmax_cross_entropy(y, t))

"""insert your training loop"""
```

### Prediction
```py
from vggface import VGGFace
import numpy as np
import cv2

image = cv2.imread("./image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))[np.newaxis].transpose(0, 3, 1, 2)/255.


model = VGGFace([1, 3, 244, 244])
model.x.d = image
model.output.forward()
output = model.output.d.copy()
print(output[0].argmax())
```

### Reference
- [VGG-Face CNN descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
