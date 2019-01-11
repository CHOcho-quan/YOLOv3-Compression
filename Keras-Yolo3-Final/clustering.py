from keras.models import Model, load_model
from sklearn.cluster import KMeans
import numpy as np
import os
from yolo import MaskedConv2D

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def weight_share(weights, bits=5):
    shape = weights.shape
    weights = weights.reshape(-1, 1)
    _max = np.max(weights)
    _min = np.min(weights)
    space = np.linspace(_min, _max, num=2**bits)
    kmeans = KMeans(n_clusters=len(space), random_state=0).fit(weights)
    return kmeans.cluster_centers_[kmeans.labels_].reshape(shape)

      
model = load_model("kangaroo.h5", custom_objects={'MaskedConv2D': MaskedConv2D})
for i in range(0, 252):
    layer = model.get_layer(index=i)
    if 'conv' in layer.name:
        print(layer.name)
        weights = layer.get_weights()
        if len(weights) > 1:
            kernel = weights[0]
            bias = weights[1]
            kernel = weight_share(kernel)
            layer.set_weights([kernel, bias])
        else:
            kernel = weights[0]
            kernle = weight_share(kernel)
            layer.set_weights([kernel])
model.save_weights("cluster.h5", overwrite=True)