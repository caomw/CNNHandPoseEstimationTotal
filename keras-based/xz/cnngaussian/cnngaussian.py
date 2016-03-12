import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from seya.layers.attention import GaussianModel,BallModel,ConvertToXY

from PIL import Image

BATCHSIZE=1
NB_EPOCH=100
HEIGHT=96
WIDTH=96
BALLNUM=48
SIGMA=30

##To 26 Degree
seq96=Sequential()
seq96.add(Convolution2D(16,1,5,5,activation='relu'))
seq96.add(MaxPooling2D(poolsize=(4,4)))
seq96.add(Convolution2D(32,16,2,2,activation='relu'))
seq96.add(MaxPooling2D(poolsize=(2,2)))
seq96.add(Flatten())
seq96.add(Dense(3872,26))

##48*3
ball=Sequential()
ball.add(BallModel(prenet=seq96,batchsize=BATCHSIZE))

##48*2 ( (xk/zk+0.5)*width  (yk/zk+0.5)*height)
xy=Sequential()
xy.add(ConvertToXY(prenet=ball,batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM))

##Gaussian Sphere Model
main=Sequential()
main.add(GaussianModel(prenet=ball,height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE))


sgd = SGD(lr=0.001,gamma=0.00005,power=0.75, momentum=0.9, nesterov=True)

main.compile(optimizer=sgd,loss='mse')

print 'Compile Done'

kind=1
seq=1     
filename="%s%d%s%d%s" % ("/home/dell/handmodel/trial96",kind,"_",seq,".h5")
X96=HDF5Matrix(filename,'data96')
data96 = np.empty(shape=X96.data.shape, dtype=X96.data.dtype)
data96[:]=X96.data[:]

minloss=1111.0
for epoch in range(0,NB_EPOCH):
  trainloss=graph.fit(data96, data96, batch_size=BATCHSIZE,nb_epoch=1)
  print trainloss
  main.save_weights('/home/dell/handmodel/mainmodel.h5',overwrite=True)
