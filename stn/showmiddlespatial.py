import numpy as np
import math
from keras.models import Graph,Sequential,model_from_json
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Dense,Activation,Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from seya.layers.attention import SpatialTransformer, ST2
from PIL import Image

#####Load Origin Model
graphorigin=Graph()

locnetorigin96 = Sequential()
locnetorigin96.add(Convolution2D(8, 1, 5, 5,activation='relu'))
locnetorigin96.add(MaxPooling2D(poolsize=(2,2)))
locnetorigin96.add(Convolution2D(12, 8, 5, 5,activation='relu'))
locnetorigin96.add(MaxPooling2D(poolsize=(2,2)))
locnetorigin96.add(Convolution2D(16,8, 4, 4,activation='relu'))
locnetorigin96.add(MaxPooling2D(poolsize=(2,2)))
locnetorigin96.add(Flatten())
locnetorigin96.add(Dense(1296,50))
locnetorigin96.add(Activation('relu'))
locnetorigin96.add(Dense(50, 6))
#####48*48 Spatial Transform Network
locnetorigin48 = Sequential()

locnetorigin48.add(Convolution2D(4,1,3,3,activation='relu'))
locnetorigin48.add(Convolution2D(8,4,3,3,activation='relu'))
locnetorigin48.add(MaxPooling2D(poolsize=(2,2)))
locnetorigin48.add(Convolution2D(12,8,3,3,activation='relu'))
locnetorigin48.add(MaxPooling2D(poolsize=(2,2)))
locnetorigin48.add(Flatten())
locnetorigin48.add(Dense(1200,50))
locnetorigin48.add(Activation('relu'))
locnetorigin48.add(Dense(50,6))
#####24*24 Spatial Transform Network
locnetorigin24 = Sequential()
locnetorigin24.add(Convolution2D(4,1,3,3,activation='relu'))
locnetorigin24.add(Convolution2D(8,4,3,3,activation='relu'))
locnetorigin24.add(MaxPooling2D(poolsize=(2,2)))
locnetorigin24.add(Flatten())
locnetorigin24.add(Dense(800,50))
locnetorigin24.add(Activation('relu'))
locnetorigin24.add(Dense(50,6))


######input 96*96
graphorigin.add_input(name='oinput96',ndim=4)
graphorigin.add_node(SpatialTransformer(localization_net=locnetorigin96, downsample_factor=1),name='input96',input='oinput96')
graphorigin.add_node(Convolution2D(16,1,5,5,activation='relu'),name='conv196',input='input96')
graphorigin.add_node(MaxPooling2D(poolsize=(4,4)),name='pool196',input='conv196')
graphorigin.add_node(Convolution2D(32,16,2,2,activation='relu'),name='conv296',input='pool196')
graphorigin.add_node(MaxPooling2D(poolsize=(2,2)),name='pool296',input='conv296')
graphorigin.add_node(Flatten(),name='flatten96',input='pool296')
graphorigin.add_node(Dense(32*11*11,6776,init='normal',activation='relu'),name='dense96',input='flatten96')
######input 48*48
graphorigin.add_input(name='oinput48',ndim=4)
graphorigin.add_node(SpatialTransformer(localization_net=locnetorigin48, downsample_factor=1),name='input48',input='oinput48')
graphorigin.add_node(Convolution2D(16,1,3,3,activation='relu'),name='conv148',input='input48')
graphorigin.add_node(MaxPooling2D(poolsize=(2,2)),name='pool148',input='conv148')
graphorigin.add_node(Convolution2D(32,16,2,2,activation='relu'),name='conv248',input='pool148')
graphorigin.add_node(MaxPooling2D(poolsize=(2,2)),name='pool248',input='conv248')
graphorigin.add_node(Flatten(),name='flatten48',input='pool248')
graphorigin.add_node(Dense(32*11*11,6776,init='normal',activation='relu'),name='dense48',input='flatten48')
######input 24*24
graphorigin.add_input(name='oinput24',ndim=4)
graphorigin.add_node(SpatialTransformer(localization_net=locnetorigin24, downsample_factor=1),name='input24',input='oinput24')
graphorigin.add_node(Convolution2D(16,1,2,2,activation='relu'),name='conv124',input='input24')
graphorigin.add_node(MaxPooling2D(poolsize=(2,2)),name='pool124',input='conv124')
graphorigin.add_node(Flatten(),name='flatten24',input='pool124')
graphorigin.add_node(Dense(16*11*11,6776,init='normal',activation='relu'),name='dense24',input='flatten24')
######Concat
graphorigin.add_node(Dense(6776,6776,init='normal',activation='relu',), name='denseall1', inputs=['dense96','dense48','dense24'],merge_mode='sum')
graphorigin.add_node(Dense(6776,6776,init='normal',activation='relu'), name='denseall2',input='denseall1')

######Compute Loss
graphorigin.add_output(name='outputall', input='denseall2')
graphorigin.load_weights('/home/zxy_wqf/keras-master/examples/cnn1/graphmodelspatial.h5')
sgd = SGD(lr=0.0001, decay=0.002, momentum=0.7, nesterov=True)
graphorigin.compile(optimizer=sgd,loss={'outputall':'mse'})

print 1
print graphorigin.nodes

#########Define New Network
graph=Graph()

locnet96 = Sequential()
locnet96.add(Convolution2D(4, 1, 5, 5,activation='relu'))
locnet96.add(MaxPooling2D(poolsize=(2,2)))
locnet96.add(Convolution2D(8, 4, 5, 5,activation='relu'))
locnet96.add(MaxPooling2D(poolsize=(2,2)))
locnet96.add(Convolution2D(16,8, 4, 4,activation='relu'))
locnet96.add(MaxPooling2D(poolsize=(2,2)))
locnet96.add(Flatten())
locnet96.add(Dense(1296, 400))
locnet96.add(Activation('relu'))
locnet96.add(Dense(400, 50))
locnet96.add(Activation('relu'))
locnet96.add(Dense(50, 6))

graph.add_input(name='oinput96',ndim=4)
graph.add_node(SpatialTransformer(localization_net=locnet96, downsample_factor=1,weights=graphorigin.nodes.values()[5].get_weights()),name='input96',input='oinput96')
graph.add_output(name='outputall', input='input96')
sgd = SGD(lr=0.005, decay=0.005, momentum=0.9, nesterov=True)
graph.compile(optimizer=sgd,loss={'outputall':'mse'})

cnt=0
for epoch in range(0,1):
    for seq in range(1,2):
        for kind in range(1,2):       
           cnt=cnt+1 	
           print epoch,kind,seq
           filename="%s%d%s%d%s" % ("/home/zxy_wqf/keras-master/examples/cnn1/trial96",kind,"_",seq,".h5")
           X96=HDF5Matrix(filename,'data96',0,12125)
           data96 = np.empty(shape=X96.data.shape, dtype=X96.data.dtype)
           data96[:]=X96.data[:]
           data96=data96[1:400]

           Y96=HDF5Matrix(filename,'label96',0,12125)
           label96 = np.empty(shape=Y96.data.shape, dtype=Y96.data.dtype)
           label96[:]=Y96.data[:]
           label96=label96[1:400]

           filename="%s%d%s%d%s" % ("/home/zxy_wqf/keras-master/examples/cnn1/trial48",kind,"_",seq,".h5")
           X48=HDF5Matrix(filename,'data48',0,12125)
           data48 = np.empty(shape=X48.data.shape, dtype=X48.data.dtype)
           data48[:]=X48.data[:]
           data48=data48[1:400]

           filename="%s%d%s%d%s" % ("/home/zxy_wqf/keras-master/examples/cnn1/trial24",kind,"_",seq,".h5")
           X24=HDF5Matrix(filename,'data24',0,12125)
           data24 = np.empty(shape=X24.data.shape, dtype=X24.data.dtype)
           data24[:]=X24.data[:]
           data24=data24[1:400]

           predictions=graph.predict({'oinput96':data96,'oinput48':data48,'oinput24':data24})
           omap=Image.new("L",(96,96))
           a=omap.load()
           #sum=0.0
           
           print predictions
           print predictions.values()[0].shape
           minv=1111 
           maxv=0
           for id in range(19,20):             
              for j in range(0,96):
                 for k in range(0,96):                      
                    a[j,k]=predictions.values()[0][id][0][j][k]*255    
           omap=omap.resize((500,500),Image.ANTIALIAS)             
           #sum=sum+math.fabs(predictions.values()[0][id][i*484+j*22+k]-label96[id,i*484+j*22+k])
           #sum=sum/6776.0/1000.0
           #print sum
           omap.show()
