import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.custom import GaussianModel,BallModel,ConvertToXY,OnlyFlatten


BATCHSIZE = 1
NB_EPOCH = 100
HEIGHT = 96
WIDTH = 96
BALLNUM = 48
SIGMA = 30

modell = Sequential()
modell.add(Convolution2D(16, 1, 5, 5, activation = 'relu'))
modell.add(MaxPooling2D(poolsize = (4, 4)))
modell.add(Convolution2D(32, 16, 2, 2, activation = 'relu'))
modell.add(MaxPooling2D(poolsize = (2, 2)))
modell.add(Flatten())


modelm=Sequential()
modelm.add(MaxPooling2D(poolsize = (2, 2)))
modelm.add(Convolution2D(16, 1, 3, 3, activation = 'relu'))
modelm.add(MaxPooling2D(poolsize = (2, 2)))
modelm.add(Convolution2D(32, 16, 2, 2, activation = 'relu'))
modelm.add(MaxPooling2D(poolsize = (2, 2)))
modelm.add(Flatten())

models=Sequential()
models.add(MaxPooling2D(poolsize = (4, 4)))
models.add(Convolution2D(32, 1, 3, 3, activation = 'relu'))
models.add(MaxPooling2D(poolsize = (2, 2)))
models.add(Flatten())

modelheat=Sequential()
modelheat.add(Merge([modell,modelm,models],mode='sum'))
modelheat.add(Dense(3872, 4096, activation = 'relu'))
modelheat.add(Dense(4096, 6776))

modelgaussian=Sequential()
modelgaussian.add(OnlyFlatten(prenet=modelheat))
modelgaussian.add(Dense(6776, 2048, activation = 'relu'))
modelgaussian.add(Dense(2048, 512, activation = 'relu'))
modelgaussian.add(Dense(512, 26, activation = 'relu'))

modelgaussian.add(BallModel(batchsize = BATCHSIZE))
modelgaussian.add(ConvertToXY(batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM))
modelgaussian.add(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE))
modelgaussian.add(Flatten())

modelmain=Sequential()
modelmain.add(Merge([modelheat,modelgaussian],mode='concat'))


sgd = SGD(lr = 0.01, decay = 0.0005, momentum = 0.9, nesterov = False)
modelgaussian.compile(optimizer = sgd, loss = 'mse')

print 'compile ok.'
print modelmain.layers


