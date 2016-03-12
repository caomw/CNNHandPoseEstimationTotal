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
from PIL import Image
sys.setrecursionlimit(132768)

BATCHSIZE = 1
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

sgd = SGD(lr = 0.00001,sigmma=0.0005,power=0.75, momentum = 0.9, nesterov =True)
minloss=0.41

model = Graph()
model.add_input(name = 'inputL', ndim = 4)
model.add_node(Convolution2D(24, 1, 5, 5, activation = 'relu'), name = 'convL1', input = 'inputL')
model.add_node(MaxPooling2D(poolsize = (4, 4)), name = 'poolL1', input = 'convL1')
model.add_node(Convolution2D(48, 24, 2, 2, activation = 'relu'), name = 'convL2', input = 'poolL1')
model.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolL2', input = 'convL2')
model.add_node(Flatten(), name = 'FL', input = 'poolL2');
"""
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
"""
model.add_node(Dense(5808,1024, init='uniform', activation = 'relu'), name = 'FL1', input='FL')
model.add_node(Dense(1024,512, init='uniform', activation = 'relu'), name = 'FL2', input='FL1')

#model.add_node(Dense(6776, 6776,init='uniform'),name = 'heatmap', input = 'concat')

#model.add_node(Dense(4096,1024, activation = 'sigmoid',init='uniform'), name = 'dense1', input = 'concat')
#model.add_node(Dense(2048, 512, activation = 'relu',init='uniform'), name = 'dense2', input = 'dense1')
model.add_node(Dense(512,26, init='uniform'), name = 'dof', input = 'FL2')

model.add_node(BallModel(height=HEIGHT,width=WIDTH,batchsize = BATCHSIZE), name = 'ball', input = 'dof')

#model.add_node(ConvertToXY(batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM),name = 'xy', input = 'ball')

model.add_node(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE), name = 'main', input = 'ball')
model.add_node(Flatten(),name='out',input='main')
#model.add_output(name = 'depth', input = 'main')
model.add_output(name='main',input='out')
model.add_output(name='doflimit',input='dof')
#model.add_output(name='heatmap',input='heatmap')

model.load_weights('/home/strawberryfg/cnngaussian/graphrender.h5')
#model.compile(optimizer = sgd, loss = {'main':forth_cost_simplify,'doflimit':degreelimit})

np.random.seed(0)
print model.nodes

print 'compile ok.'


modeltest = Graph()
modeltest.add_input(name = 'inputL', ndim = 4)
modeltest.add_node(Convolution2D(24, 1, 5, 5, activation = 'relu',weights=model.nodes.values()[2].get_weights()), name = 'convL1', input = 'inputL')
modeltest.add_node(MaxPooling2D(poolsize = (4, 4)), name = 'poolL1', input = 'convL1')
modeltest.add_node(Convolution2D(48, 24, 2, 2, activation = 'relu',weights=model.nodes.values()[3].get_weights()), name = 'convL2', input = 'poolL1')
modeltest.add_node(MaxPooling2D(poolsize = (2, 2)), name = 'poolL2', input = 'convL2')
modeltest.add_node(Flatten(), name = 'FL', input = 'poolL2');
"""
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
"""
modeltest.add_node(Dense(5808,1024, init='uniform', activation = 'relu',weights=model.nodes.values()[1].get_weights()), name = 'FL1', input='FL')
modeltest.add_node(Dense(1024,512, init='uniform', activation = 'relu',weights=model.nodes.values()[4].get_weights()), name = 'FL2', input='FL1')

#model.add_node(Dense(6776, 6776,init='uniform'),name = 'heatmap', input = 'concat')

#model.add_node(Dense(4096,1024, activation = 'sigmoid',init='uniform'), name = 'dense1', input = 'concat')
#model.add_node(Dense(2048, 512, activation = 'relu',init='uniform'), name = 'dense2', input = 'dense1')
modeltest.add_node(Dense(512,26, init='uniform',weights=model.nodes.values()[5].get_weights()), name = 'dof', input = 'FL2')

modeltest.add_node(BallModel(height=HEIGHT,width=WIDTH,batchsize = BATCHSIZE), name = 'ball', input = 'dof')

#model.add_node(ConvertToXY(batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM),name = 'xy', input = 'ball')

modeltest.add_node(GaussianModel(height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE), name = 'main', input = 'ball')
modeltest.add_node(Flatten(),name='out',input='main')
#model.add_output(name = 'depth', input = 'main')
modeltest.add_output(name='main',input='out')
print 'now'
modeltest.compile(optimizer = sgd, loss = {'main':'mse'})


data96 = np.zeros((1000,1,96,96),dtype='float32')

#label = np.zeros((72756,6776),dtype='float32')
for potionid in range(2,3):
   filename="%s%d%s" % ("/home/strawberryfg/cnngaussian/trial961_",potionid,".h5")
   X96=HDF5Matrix(filename,'data96ori',0,12125)             
   
   data96[0:1000]=X96.data[0:1000]
   
  
   #Y=HDF5Matrix(filename,'label96',0,12125)              
   #label[st:en]=Y.data[:]
"""


for id in range(0,20):
   print 'id:'
   for i in range(0,24):
      print predictions[id][2*i],predictions[id][2*i+1],
   print "\n"


"""

"""
for id in range(0,20):
   print 'id:'
   for i in range(0,13):
      print predictions[id][2*i],predictions[id][2*i+1],
   print "\n"
"""
predictions=modeltest.predict({'inputL':data96},batch_size=BATCHSIZE)
predictions=predictions['main']
for id in range(0,1000):
  omap=Image.new("L",(96,96))
  a=omap.load()
  for i in range(0,96):
    for j in range(0,96):
      a[j,i]=predictions[id][i*96+j]*255
  omap=omap.resize((500,500),Image.ANTIALIAS)
  filename='/home/strawberryfg/cnngaussian/map/'+str(id)+'.png'
  omap.save(filename)