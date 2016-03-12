import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.customnew import GaussianModel,BallModel
import sys
import theano.tensor as T
sys.setrecursionlimit(132768)

BATCHSIZE = 2
NB_EPOCH = 100
HEIGHT = 96
WIDTH = 96
BALLNUM = 48
SIGMA = 50

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
   
   fz=((y_true + y_pred -2*y_true*y_pred)).sum(axis=[1])
   fm=(y_true + y_pred).sum(axis=[1])
   ret=(fz/fm)
   return ret

def degreelimit(y_true,y_pred):
   limitlow=np.array([-1.0,-1.0,-1.0,
                      -math.pi/4,-math.pi/2,-math.pi/2,
                      0.0,0.0,0.0,0.0,
                      -math.pi/9,-math.pi/9,0.0,0.0,
                      -math.pi/9,-math.pi/18,0.0,0.0,
                      -math.pi/9,-math.pi/18,0.0,0.0,
                      -math.pi/9,-math.pi/18,0.0,0.0])
   limitup=np.array([1.0,1.0,1.0,
                     math.pi/2,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/2,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/18,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/18,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/18,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/9,math.pi/2,math.pi/2])
   low=theano.shared(limitlow)
   low=low.reshape((1,26))
   up=theano.shared(limitup)
   up=up.reshape((1,26))
   low=T.tile(low,(BATCHSIZE,1))
   up=T.tile(up,(BATCHSIZE,1))
   ret=(T.maximum(y_pred,up)-up)+(low-T.minimum(low,y_pred))
   ret=ret.mean(axis=1)
   return ret

doflimitlow=np.array([-1.0,-1.0,-1.0,
                      -math.pi/4,-math.pi/2,-math.pi/2,
                      0.0,0.0,0.0,0.0,
                      -math.pi/9,-math.pi/9,0.0,0.0,
                      -math.pi/9,-math.pi/18,0.0,0.0,
                      -math.pi/9,-math.pi/18,0.0,0.0,
                      -math.pi/9,-math.pi/18,0.0,0.0])
doflimitup=np.array([1.0,1.0,1.0,
                     math.pi/2,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/2,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/18,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/18,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/18,math.pi/2,math.pi/2,
                     math.pi/2,math.pi/9,math.pi/2,math.pi/2])
base_lr=0.01
sgd = SGD(lr = base_lr,sigmma=0.00005,power=0.75, momentum = 0.9, nesterov =True)
minloss=0.4333

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

model.add_node(Dense(9680,4096, init='uniform', activation = 'relu'), name = 'FL1', input='densein')
model.add_node(Dense(4096,1024, init='uniform', activation = 'relu'), name = 'FL2', input='FL1')
model.add_node(Dense(1024,128, init='uniform', activation = 'relu'), name = 'FL3', input='FL2')

model.add_node(Dense(128,26, init='uniform'), name = 'dof', input = 'FL3')

model.add_node(BallModel(height=HEIGHT,width=WIDTH,batchsize = BATCHSIZE), name = 'ball', input = 'dof')


model.add_node(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE), name = 'main', input = 'ball')
model.add_node(Flatten(),name='out',input='main')

model.add_output(name='main',input='out')
model.add_output(name='doflimit',input='dof')

model.load_weights('/home/strawberryfg/cnngaussian/new/graphrender.h5')
model.compile(optimizer = sgd, loss = {'main':forth_cost_simplify,'doflimit':degreelimit})
np.random.seed(0)
print model.nodes

print 'compile ok.'
x=np.zeros((40,1,96,96),dtype='float32')
xori=np.zeros((40,9216),dtype='float32')
limits=np.zeros((40,26),dtype='float32')


ind=np.zeros((72960),dtype='int32')
for i in range(0,72960):
   ind[i]=i

for kind in range(1,2):      #1-5
   for potionid in range(1,7):          #1-6  data batch 12126      
      
      print kind,potionid


data96 = np.zeros((72960,1,96,96),dtype='float32')
data96ori = np.zeros((72960,1,96,96),dtype='float32')

for potionid in range(1,7):
   filename="%s%d%s" % ("/home/strawberryfg/cnngaussian/trial961_",(kind-1)*6+potionid,".h5")
   X96=HDF5Matrix(filename,'data96ori',0,12125)             
   st=(potionid-1)*12126
   en=potionid*12126
   data96[st:en]=X96.data[:]

   X96ori=HDF5Matrix(filename,'data96',0,12125)              
   data96ori[st:en]=X96ori.data[:]

data96ori=data96ori.reshape((72960,9216))
for i in range(0,204):
   data96[i+72756]=data96[i]
   data96ori[i+72756]=data96ori[i]

for epoch in range(0,1000):   
   np.random.shuffle(ind)
   
   for potionid in range(1,456*4+1):
      
      for i in range(0,40):
         x[i]=data96[ind[(potionid-1)*40+i]]
         xori[i]=data96ori[ind[(potionid-1)*40+i]]
         
      
      flag=1
      cnt=0
      flagnan=0
      while flag==1:
         print epoch,' ',potionid,' ',cnt
         cnt=cnt+1

         his=model.fit({'inputL':x,'main':xori,'doflimit':limits},batch_size=BATCHSIZE,nb_epoch=1,shuffle=True)
         if math.isnan(his.history['loss'][0])==True:
            flagnan=1
            break
         else:
            flag=0
         if cnt>3:
            break
         """
         predicts=model.predict({'inputL':x},batch_size=BATCHSIZE)
         predictionslimit=predicts['doflimit']        
         sum=0
         for id in range(0,40):
            sumnow=0
            for i in range(0,26):
               sumnow=sumnow+max(predictionslimit[id][i],doflimitup[i])-doflimitup[i]+doflimitlow[i]-min(doflimitlow[i],predictionslimit[id][i])
            sumnow=sumnow/26
            sum=sum+sumnow
         sum=sum/40
         print 'doflimit:',sum,' forth_loss:',his.history['loss'][0]-sum
         """
         if his.history['loss'][0]<minloss:
            minloss=his.history['loss'][0]
            
            print 'nowloss:',his.history['loss'][0]
            print 'minloss:',minloss
            model.save_weights('/home/strawberryfg/cnngaussian/new/graphrender.h5',overwrite=True)
         print his.history
         model.save_weights('/home/strawberryfg/cnngaussian/new/graphrender2.h5',overwrite=True)
      if flagnan==1:
         model.load_weights('/home/strawberryfg/cnngaussian/new/graphrender.h5')
         base_lr=max(base_lr*0.9,0.0000000001)
         sgd = SGD(lr = base_lr,sigmma=0.0005,power=0.75, momentum = 0.9, nesterov =True)
         model.compile(optimizer = sgd, loss = {'main':forth_cost_simplify,'doflimit':degreelimit})