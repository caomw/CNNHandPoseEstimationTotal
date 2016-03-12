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


graphalready=Graph()

######input 96*96
graphalready.add_input(name='oinput96',ndim=4)

graphalready.add_node(Convolution2D(16,1,5,5,activation='relu'),name='conv196',input='oinput96')
graphalready.add_node(MaxPooling2D(poolsize=(4,4)),name='pool196',input='conv196')
graphalready.add_node(Convolution2D(32,16,2,2,activation='relu'),name='conv296',input='pool196')
graphalready.add_node(MaxPooling2D(poolsize=(2,2)),name='pool296',input='conv296')
graphalready.add_node(Flatten(),name='flatten96',input='pool296')
graphalready.add_node(Dense(32*11*11,6776,init='normal',activation='relu'),name='dense96',input='flatten96')
######input 48*48
graphalready.add_input(name='oinput48',ndim=4)

graphalready.add_node(Convolution2D(16,1,3,3,activation='relu'),name='conv148',input='oinput48')
graphalready.add_node(MaxPooling2D(poolsize=(2,2)),name='pool148',input='conv148')
graphalready.add_node(Convolution2D(32,16,2,2,activation='relu'),name='conv248',input='pool148')
graphalready.add_node(MaxPooling2D(poolsize=(2,2)),name='pool248',input='conv248')
graphalready.add_node(Flatten(),name='flatten48',input='pool248')
graphalready.add_node(Dense(32*11*11,6776,init='normal',activation='relu'),name='dense48',input='flatten48')
######input 24*24
graphalready.add_input(name='oinput24',ndim=4)

graphalready.add_node(Convolution2D(16,1,2,2,activation='relu'),name='conv124',input='oinput24')
graphalready.add_node(MaxPooling2D(poolsize=(2,2)),name='pool124',input='conv124')
graphalready.add_node(Flatten(),name='flatten24',input='pool124')
graphalready.add_node(Dense(16*11*11,6776,init='normal',activation='relu'),name='dense24',input='flatten24')
######Concat
graphalready.add_node(Dense(6776,6776,init='normal',activation='relu',), name='denseall1', inputs=['dense96','dense48','dense24'],merge_mode='sum')
graphalready.add_node(Dense(6776,6776,init='normal',activation='relu'), name='denseall2',input='denseall1')
######Compute Loss
graphalready.add_output(name='outputall', input='denseall2')

graphalready.load_weights('/home/zxy_wqf/keras-master/examples/cnn1/graphmodel.h5')

sgd = SGD(lr=0.005, decay=0.005, momentum=0.9, nesterov=True)
graphalready.compile(optimizer=sgd,loss={'outputall':'mse'})

print 1
print graphalready.nodes
graph=Graph()

######input 48*48
graph.add_input(name='oinput48',ndim=4)

graph.add_node(Convolution2D(16,1,3,3,activation='relu',weights=graphalready.nodes.values()[5].get_weights()),name='conv148',input='oinput48')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool148',input='conv148')
graph.add_node(Convolution2D(32,16,2,2,activation='relu',weights=graphalready.nodes.values()[3].get_weights()),name='conv248',input='pool148')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool248',input='conv248')
graph.add_node(Flatten(),name='flatten48',input='pool248')
graph.add_node(Dense(32*11*11,6776,init='normal',activation='relu',weights=graphalready.nodes.values()[13].get_weights()),name='dense48',input='flatten48')

graph.add_output(name='outputall', input='dense48')

#graph.load_weights('/home/zxy_wqf/keras-master/examples/cnn1/graphmodel.h5')

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
           heatmap=Image.new("L",(22,22))
           a=heatmap.load()
           #sum=0.0
           
           minx=111111.0
           maxx=0.0
           for id in range(159,160):
              for  i in range(8,9):
                 for j in range(0,22):
                    for k in range(0,22):
                        if  predictions.values()[0][id][i*484+j*22+k]>1e-6:
                            minx=min(minx,predictions.values()[0][id][i*484+j*22+k])	
                            maxx=max(maxx,predictions.values()[0][id][i*484+j*22+k])
           for id in range(159,160):
              for  i in range(8,9):
                  for j in range(0,22):
                     for k in range(0,22):
                        if  predictions.values()[0][id][i*484+j*22+k]>1e-6:
                            a[j,k]=(predictions.values()[0][id][i*484+j*22+k]-minx)/(maxx-minx)*255.0
                        else:
                        	 a[j,k]=0
           print minx,maxx
           heatmap=heatmap.resize((500,500),Image.ANTIALIAS)             
           #sum=sum+math.fabs(predictions.values()[0][id][i*484+j*22+k]-label96[id,i*484+j*22+k])
           #sum=sum/6776.0/1000.0
           #print sum
           heatmap.show()