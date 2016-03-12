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
###96*96
graph=Graph()
graph.add_input(name='data96',ndim=4)
graph.add_node(Convolution2D(16,1,5,5,activation='relu'),name='conv196',input='data96')
graph.add_node(MaxPooling2D(poolsize=(4,4)),name='pool196',input='conv196')
graph.add_node(Convolution2D(32,16,2,2,activation='relu'),name='conv296',input='pool196')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool296',input='conv296')
graph.add_node(Flatten(),name='flatten96',input='pool296')
###48*48
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='data48',input='data96')
graph.add_node(Convolution2D(16,1,3,3,activation='relu'),name='conv148',input='data48')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool148',input='conv148')
graph.add_node(Convolution2D(32,16,2,2,activation='relu'),name='conv248',input='pool148')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool248',input='conv248')
graph.add_node(Flatten(),name='flatten48',input='pool248')
###24*24
graph.add_node(MaxPooling2D(poolsize=(4,4)),name='data24',input='data96')
graph.add_node(Convolution2D(16,1,3,3,activation='relu'),name='conv124',input='data24')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool124',input='conv124')
graph.add_node(Flatten(),name='flatten24',input='pool124')
##Concat
graph.add_node(Flatten(),name='densein',inputs=['flatten96','flatten48','flatten24'],merge_mode='concat',concat_axis=1)
graph.add_node(Dense(9680,6776,init='normal',activation='relu'),name='denseall1',input='densein')
graph.add_node(Dense(6776,6776,init='normal'),name='denseall2',input='denseall1')
graph.add_node(Dense(6776,2048,init='normal',activation='relu'),name='denseall3',input='denseall2')
graph.add_node(Dense(2048,512,init='normal',activation='relu'),name='denseall4',input='denseall3')
graph.add_node(Dense(512,26,init='normal',activation='relu'),name='denseall5',input='denseall4')

graph.add_node(BallModel(batchsize=BATCHSIZE),name='ballxyz',input='denseall5')
#graph.add_node(ConvertToXY(batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM),name='ballxy',input='ballxyz')
#graph.add_node(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE),name='gaussian',input='ballxy')
#graph.add_output(name='heatmap',input='denseall2')
#graph.add_output(name='gaussianmap',input='gaussian')
graph.add_output(name='ball',input='ballxyz')
##To 26 Degree
print graph.nodes
print graph.get_output()
"""
##48*3
ball=Sequential()
ball.add(BallModel(prenet=graph,batchsize=BATCHSIZE))

##48*2 ( (xk/zk+0.5)*width  (yk/zk+0.5)*height)
xy=Sequential()
xy.add(ConvertToXY(prenet=ball,batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM))

##Gaussian Sphere Model
main=Sequential()
main.add(GaussianModel(prenet=ball,height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE))
"""

sgd = SGD(lr=0.5, momentum=0.9, nesterov=True)

graph.compile(optimizer=sgd,loss={'ball':'mse'})

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
