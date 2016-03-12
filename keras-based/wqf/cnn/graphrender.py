import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.customfinal import GaussianModel,BallModel
import sys
import theano.tensor as T
sys.setrecursionlimit(132768)

"""
def mean_squared_error_depth(y_true, y_pred):
    return 0.01*T.sqr(y_pred - y_true).mean(axis=-1)
"""
def mean_squared_error_per_picture(y_true,y_pred):
   allpic=T.sqr(y_pred - y_true).sum(axis=[1]) ##sigma (  () ^2) [ ]  batchsize 
   perpic=T.mean(allpic/2.0,axis=-1)
   return perpic

def forth_cost_simplify(y_true, y_pred):
   intersect=y_true*y_pred
   combine=y_true+y_pred-intersect
   ret=((combine-intersect)/(combine+intersect)).mean(axis=-1)
   return ret


BATCHSIZE = 128
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
model.add_node(Dense(9680, 6776,init='uniform', activation = 'relu'), name = 'concat', input='densein')
model.add_node(Dense(6776, 6776,init='uniform'),name = 'heatmap', input = 'concat')

sgd = SGD(lr = 0.0000001,sigmma=0.0000005,power=0.75, momentum = 0.9, nesterov = True)

model.add_node(Dense(6776, 1024, activation = 'relu',init='uniform'), name = 'dense1', input = 'heatmap')
#model.add_node(Dense(2048, 512, activation = 'relu',init='uniform'), name = 'dense2', input = 'dense1')
model.add_node(Dense(1024, 26, activation = 'relu',init='uniform'), name = 'dof', input = 'dense1')

model.add_node(BallModel(height=HEIGHT,width=WIDTH,batchsize = BATCHSIZE), name = 'ball', input = 'dof')

#model.add_node(ConvertToXY(batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM),name = 'xy', input = 'ball')

model.add_node(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE), name = 'main', input = 'ball')

#model.add_output(name = 'depth', input = 'main')
model.add_output(name='main',input='main')
model.add_output(name='heatmap',input='heatmap')
model.compile(optimizer = sgd, loss = {'main':forth_cost_simplify,'heatmap':mean_squared_error_per_picture})

print model.nodes

print 'compile ok.'
x=np.zeros((12128,1,96,96),dtype='float32')
xori=np.zeros((12128,1,96,96),dtype='float32')
y=np.zeros((12128,6776),dtype='float32')

xx=np.zeros((256,1,96,96),dtype='float32')
xxori=np.zeros((256,1,96,96),dtype='float32')
yy=np.zeros((256,6776),dtype='float32')
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
         np.random.shuffle(ind)

         filename="%s%d%s" % ("/home/strawberryfg/cnngaussian/trial961_",(kind-1)*6+potionid,".h5")
         X96=HDF5Matrix(filename,'data96',0,12125)        
         data96 = np.empty(shape=X96.data.shape, dtype="float32")
         data96[:]=X96.data[:]


         X96ori=HDF5Matrix(filename,'data96ori',0,12125)        
         data96ori = np.empty(shape=X96ori.data.shape, dtype="float32")
         data96ori[:]=X96ori.data[:]

         Y96=HDF5Matrix(filename,'label96',0,12125)
         label96 = np.empty(shape=Y96.data.shape, dtype="float32")
         label96[:]=Y96.data[:]

         
         for i in range(0,12126):
            x[i]=data96[ind[i]]
            y[i]=label96[ind[i]]
            xori[i]=data96ori[ind[i]]

         x[12126]=x[0]
         x[12127]=x[1]
         y[12126]=y[0]
         y[12127]=y[1]
         xori[12126]=xori[0]
         xori[12127]=xori[1]

         #x=x[0:512]
         #xori=xori[0:512]
         #y=y[0:512]
         his=model.fit({'inputL':x,'main':xori,'heatmap':y},batch_size=BATCHSIZE,nb_epoch=1,shuffle=True)
         print his.history
         if his.history['loss'][0]<minloss:
            minloss=his.history['loss'][0]
            model.save_weights('/home/strawberryfg/cnngaussian/graphrender.h5',overwrite=True)
         print 'nowloss:',his.history['loss'][0]
         print 'minloss:',minloss

         xx=x[0:256]
         yy=y[0:256]
         xxori=xori[0:256]
         predicts=model.predict({'inputL':xx,'main':xxori,'heatmap':yy},batch_size=4)
         main=predicts['main']
         heatmap=predicts['heatmap']

         lossmain=0.0
         for id in range(0,256):
            tot=0.0
            for i in range(0,96):
               for j in range(0,96):
                  tot=tot+ 1.0-2.0*(main[id][0][i][j]*xxori[id][0][i][j])/(main[id][0][i][j]+xxori[id][0][i][j])
            tot=tot/(96*96)
            lossmain=lossmain+tot
         lossmain=lossmain/256

         lossheat=0.0
         for id in range(0,256):
            tot=0.0
            for i in range(0,6776):
               tot=tot+(heatmap[id][i]-yy[id][i])*(heatmap[id][i]-yy[id][i])            
            lossheat=lossheat+tot/2.0
         lossheat=lossheat/256

         print 'lossmain:',lossmain,'     lossheat:',lossheat
