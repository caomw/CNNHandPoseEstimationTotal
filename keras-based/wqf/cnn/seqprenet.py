import numpy as np
import math
import theano
from keras.models import Graph,Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSample2D
from keras.layers.core import Dense,Activation,Flatten,Merge,Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.layers.customprenet import GaussianModel,BallModel,ConvertToXY,OnlyFlatten


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
modell.add(Dense(3872, 96, activation = 'relu'))
"""
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

#modeldeg=Sequential()
#modeldeg.add(OnlyFlatten(prenet=modelheat))
modelheat.add(Dense(6776, 2048, activation = 'relu'))
modelheat.add(Dense(2048, 512, activation = 'relu'))
modelheat.add(Dense(512, 26, activation = 'relu'))
#print modelheat.get_output()
"""
#ball=Sequential()
#ball.add(BallModel(prenet=modell,batchsize=BATCHSIZE))

#xy=Sequential()
#xy.add(ConvertToXY(prenet=ball,batchsize=BATCHSIZE,height=HEIGHT,width=WIDTH,ballnum=BALLNUM))

modelgaussian=Sequential()
modelgaussian.add(GaussianModel(prenet=modell,height=HEIGHT,width=WIDTH,ballnum=BALLNUM,sigma=SIGMA,batchsize=BATCHSIZE))
#modelgaussian.add(Flatten())

#modelmain=Sequential()
#modelmain.add(Merge([modelheat,modelgaussian],mode='concat'))

#print modelmain
#print modelmain.layers
sgd = SGD(lr = 0.001, sigma=0.0005,power=0.75, momentum = 0.9, nesterov = True)
modelgaussian.compile(optimizer = sgd, loss = 'mse')

x=np.zeros((12126,1,96,96),dtype='float32')
xx=np.zeros((12126,1,96,96),dtype='float32')
ind=np.zeros((12126),dtype='int32')
for potionid in range(1,2):
    filename="%s%d%s" % ("/home/strawberryfg/cnngaussian/trial961_",potionid,".h5")
    X96=HDF5Matrix(filename,'data96',0,12126,None)
    data96=np.empty(shape=X96.data.shape,dtype='float32')
    data96[:]=X96.data[:]
    x[12126*(potionid-1):12126*potionid]=data96[:] 
 
for epoch in range(0,1000):
	np.random.shuffle(ind)

	for i in range(0,12126):
		xx[i]=x[ind[i]]
	modelgaussian.fit(xx,xx,batch_size=BATCHSIZE,nb_epoch=1,shuffle=True)
print 'compile ok.'
