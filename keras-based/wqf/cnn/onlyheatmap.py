import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.customtest import GaussianModel,BallModel,ConvertToXY
import sys
sys.setrecursionlimit(32768)

BATCHSIZE = 16
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
model.add_node(Dense(9680, 6776,init='uniform', activation = 'relu'), name = 'concat', input='densein' )
model.add_node(Dense(6776, 6776, init='uniform'),name = 'heatmap', input = 'concat', create_output = True)

sgd = SGD(lr = 0.1,sigmma=0.00005,power=0.75, momentum = 0.9, nesterov = True)


model.compile(optimizer = sgd, loss = {'heatmap':'mse'})

print model.nodes

print 'compile ok.'
x=np.zeros((12126,1,96,96),dtype='float32')
y=np.zeros((12126,6776),dtype='float32')

ind=np.zeros((12126),dtype='int32')
for i in range(0,12126):
   ind[i]=i

for kind in range(1,2):      #1-5
   for potionid in range(1,7):          #1-6  data batch 12126      
      
      print kind,potionid
      
minloss=1111.0
for epoch in range(0,1000):
   for kind in range(1,2):      #1-5         
      for potionid in range(1,2):
         #np.random.shuffle(ind)

         filename="%s%d%s" % ("/home/strawberryfg/cnngaussian/trial961_",(kind-1)*6+potionid,".h5")
         X96=HDF5Matrix(filename,'data96',0,12125)        
         data96 = np.empty(shape=X96.data.shape, dtype="float32")
         data96[:]=X96.data[:]

         Y96=HDF5Matrix(filename,'label96',0,12125)
         label96 = np.empty(shape=Y96.data.shape, dtype="float32")
         label96[:]=Y96.data[:]

         for i in range(0,12126):
            x[i]=data96[ind[i]]
            y[i]=label96[ind[i]]
         loss=model.fit({'inputL':x,'heatmap':y},batch_size=BATCHSIZE,nb_epoch=1,shuffle=True)
         print loss
         if loss<minloss:
            minloss=loss
            model.save_weights('/home/strawberryfg/cnngaussian/onlyheatmap.h5')

