import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.customexpandfor import GaussianModel,BallModel
import sys
import theano.tensor as T
sys.setrecursionlimit(132768)

BATCHSIZE = 40
NB_EPOCH = 100
HEIGHT = 96
WIDTH = 96
BALLNUM = 48
SIGMA = 70

"""
def mean_squared_error_depth(y_true, y_pred):
    return 0.01*T.sqr(y_pred - y_true).mean(axis=-1)
"""
def mean_squared_error_per_picture(y_true,y_pred):
   allpic=T.sqr(y_pred - y_true).sum(axis=1) ##sigma (  () ^2) [ ]  batchsize 
   perpic=T.mean(allpic/2.0,axis=-1)
   return perpic
"""
def forth_cost_simplify(y_true, y_pred):
   intersect=y_true*y_pred
   combine=y_true+y_pred-intersect
   ret=((combine-intersect)).mean(axis=-1)
   return ret
"""
def forth_cost_simplify(y_true, y_pred):
   y_true=y_true.reshape((BATCHSIZE,1,HEIGHT,WIDTH))
   y_pred=y_pred.reshape((BATCHSIZE,1,HEIGHT,WIDTH))
   fz=((y_true + y_pred -2*y_true*y_pred)).sum(axis=[1,2,3])
   fm=(y_true + y_pred).sum(axis=[1,2,3])
   ret=((fz/fm).mean(axis=-1))
   return 0.1*ret




sgd = SGD(lr = 0.000001,sigmma=0.000005,power=0.75, momentum = 0.9, nesterov = True)
minloss=180.0

model = Graph()
model.add_input(name = 'inputL', ndim = 4)
model.add_node(Convolution2D(16, 1, 5, 5, activation = 'relu'), name = 'convL1', input = 'inputL')
model.add_node(MaxPooling2D(poolsize = (4, 4)), name = 'poolL1', input = 'convL1')
model.add_node(Convolution2D(32, 16, 2, 2, activation = 'relu'), name = 'convL2', input = 'poolL1')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolL2', input = 'convL2')
model.add_node(Flatten(), name = 'FL', input = 'poolL2');

model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'inputM', input = 'inputL')
model.add_node(Convolution2D(16, 1, 3, 3, activation = 'relu'), name = 'convM1', input = 'inputM')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolM1', input = 'convM1')
model.add_node(Convolution2D(32, 16, 2, 2, activation = 'relu'), name = 'convM2', input = 'poolM1')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolM2', input = 'convM2')
model.add_node(Flatten(), name = 'FM', input = 'poolM2')

model.add_node(MaxPooling2D(poolsize = (4, 4)), name = 'inputS', input = 'inputL')
model.add_node(Convolution2D(16, 1, 3, 3, activation = 'relu'), name = 'convS', input = 'inputS')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolS', input = 'convS')
model.add_node(Flatten(), name = 'FS', input = 'poolS');

model.add_node(Flatten(),name='densein',inputs=['FL','FM','FS'],merge_mode='concat',concat_axis=1)
model.add_node(Dense(9680, 4096,init='uniform', activation = 'relu'), name = 'concat', input='densein')
#model.add_node(Dense(6776, 6776,init='uniform'),name = 'heatmap', input = 'concat')

model.add_node(Dense(4096, 1024, activation = 'relu',init='uniform'), name = 'dense1', input = 'concat')
#model.add_node(Dense(2048, 512, activation = 'relu',init='uniform'), name = 'dense2', input = 'dense1')
model.add_node(Dense(1024, 26, init='uniform'), name = 'dof', input = 'dense1')

model.add_node(BallModel(height=HEIGHT,width=WIDTH,batchsize = BATCHSIZE), name = 'ball', input = 'dof')

#model.add_node(ConvertToXY(batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM),name = 'xy', input = 'ball')

model.add_node(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE), name = 'main', input = 'ball')

#model.add_output(name = 'depth', input = 'main')
model.add_output(name='main',input='main')
model.add_output(name='dof',input='dof')
#model.add_output(name='heatmap',input='heatmap')
#model.load_weights('/home/strawberryfg/cnngaussian/dofweights.h5')

model.compile(optimizer = sgd, loss = {'dof':'mse','main':forth_cost_simplify})
print model.nodes

print 'compile ok.'
x=np.zeros((2400,1,96,96),dtype='float32')
xori=np.zeros((2400,1,96,96),dtype='float32')
y=np.zeros((2400,26),dtype='float32')

xx=np.zeros((240,1,96,96),dtype='float32')

ind=np.zeros((2400),dtype='int32')
for i in range(0,2400):
   ind[i]=i



data96 = np.zeros((2400,1,96,96),dtype='float32')
data96ori = np.zeros((2400,1,96,96),dtype='float32')
label = np.zeros((2400,26),dtype='float32')
for potionid in range(1,2):
   filename="/home/strawberryfg/cnngaussian/degree.h5"
   X96=HDF5Matrix(filename,'data96',0,2400)             
   
   data96[:]=X96.data[:]

   X96ori=HDF5Matrix(filename,'data96ori',0,2400)              
   data96ori[:]=X96ori.data[:]
 
   Y=HDF5Matrix(filename,'labeldof',0,2400)              
   label[:]=Y.data[:]

for epoch in range(0,1000):   
   np.random.shuffle(ind)
   for potionid in range(1,2):
      for i in range(0,2400):
         x[i]=data96[ind[i]]
         xori[i]=data96ori[ind[i]]
         y[i]=label[ind[i]]
      
      his=model.fit({'inputL':x,'dof':y,'main':xori},batch_size=BATCHSIZE,nb_epoch=1,shuffle=True)
      xx[:]=x[0:240]
      
      predictions=model.predict({'inputL':xx},batch_size=BATCHSIZE)

      predictdof=predictions['dof']
      dofloss=0
      for id in range(0,240):
         tot=0
         for i in range(0,26):
            tot=tot+(predictdof[id][i]-y[id][i])*(predictdof[id][i]-y[id][i])
         tot=tot/26
         dofloss=dofloss+tot
      dofloss=dofloss/240;
      print 'dofloss:',dofloss
      print 'depthloss', his.history['loss'][0]-dofloss
      print his.history
      if his.history['loss'][0]<minloss:
         minloss=his.history['loss'][0]
         model.save_weights('/home/strawberryfg/cnngaussian/dofweights.h5',overwrite=True)
         print 'nowloss:',his.history['loss'][0]
         print 'minloss:',minloss
      
