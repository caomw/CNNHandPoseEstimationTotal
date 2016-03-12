import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.custom import GaussianModel,BallModel,ConvertToXY


BATCHSIZE = 1
NB_EPOCH = 100
HEIGHT = 96
WIDTH = 96
BALLNUM = 48
SIGMA = 30

model = Graph()
model.add_input(name = 'inputL', ndim = 4)
model.add_node(Convolution2D(16, 1, 5, 5, activation = 'relu'), name = 'convL1', input = 'inputL')
model.add_node(MaxPooling2D(poolsize = (4, 4)), name = 'poolL1', input = 'convL1')
model.add_node(Convolution2D(32, 16, 2, 2, activation = 'relu'), name = 'convL2', input = 'poolL1')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolL2', input = 'convL2')
model.add_node(Flatten(), name = 'FL', input = 'convL2');

model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'inputM', input = 'inputL')
model.add_node(Convolution2D(16, 1, 3, 3, activation = 'relu'), name = 'convM1', input = 'inputM')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolM1', input = 'convM1')
model.add_node(Convolution2D(32, 16, 2, 2, activation = 'relu'), name = 'convM2', input = 'poolM1')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolM2', input = 'convM2')
model.add_node(Flatten(), name = 'FM', input = 'convM2')

model.add_node(MaxPooling2D(poolsize = (4, 4)), name = 'inputS', input = 'inputL')
model.add_node(Convolution2D(32, 1, 3, 3, activation = 'relu'), name = 'convS', input = 'inputS')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolS', input = 'convS')
model.add_node(Flatten(), name = 'FS', input = 'convS');

model.add_node(Dense(3872, 4096, activation = 'relu'), name = 'concat', inputs = ['FL', 'FM', 'FS'], merge_mode = 'sum')
model.add_node(Dense(4096, 6776), name = 'heatmap', input = 'concat', create_output = True)

sgd = SGD(lr = 0.01, decay = 0.0005, momentum = 0.9, nesterov = False)

model.add_node(Dense(6776, 2048, activation = 'relu'), name = 'dense1', input = 'heatmap')
model.add_node(Dense(2028, 512, activation = 'relu'), name = 'dense2', input = 'dense1')
model.add_node(Dense(512, 26, activation = 'relu'), name = 'dof', input = 'dense2')



model.add_node(BallModel(prenet = ))



model.compile(optimizer = sgd, loss = {'heatmap': 'mse'})

print 'compile ok.'
