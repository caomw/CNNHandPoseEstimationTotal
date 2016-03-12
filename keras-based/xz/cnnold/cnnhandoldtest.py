import numpy as np
import math
import theano
from keras.models import Graph,Sequential,model_from_json
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Dense,Activation,Flatten,Merge
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.regularizers import l2
from PIL import Image
"""
graph=Graph()

######input 96*96
graph.add_input(name='oinput96',ndim=4)
graph.add_node(Convolution2D(24,1,5,5,activation='relu'),name='conv196',input='oinput96')
graph.add_node(MaxPooling2D(poolsize=(4,4)),name='pool196',input='conv196')
graph.add_node(Convolution2D(48,24,2,2,activation='relu'),name='conv296',input='pool196')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool296',input='conv296')
graph.add_node(Flatten(),name='flatten96',input='pool296')

######input 48*48
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='input48',input='oinput96')
graph.add_node(Convolution2D(8,1,3,3,activation='relu'),name='conv148',input='input48')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool148',input='conv148')
graph.add_node(Convolution2D(16,8,2,2,activation='relu'),name='conv248',input='pool148')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool248',input='conv248')
graph.add_node(Flatten(),name='flatten48',input='pool248')
######input 24*24
graph.add_node(MaxPooling2D(poolsize=(4,4)),name='input24',input='oinput96')
graph.add_node(Convolution2D(16,1,2,2,activation='relu'),name='conv124',input='input24')
graph.add_node(MaxPooling2D(poolsize=(2,2)),name='pool124',input='conv124')
graph.add_node(Flatten(),name='flatten24',input='pool124')

######Concat
#graph.add_node(Merge(['flatten96','flatten48','flatten24'],mode='concat'),name='merge1',inputs=['flatten96','flatten48','flatten24'])
#graph.add_node(Flatten(),name='densein',inputs=['flatten96','flatten48','flatten24'],merge_mode='concat',concat_axis=1)
#graph.add_node(Dense(6776,6776,init='normal',activation='relu',), name='denseall1', inputs=['dense96','dense48','dense24'],merge_mode='sum')
#graph.add_node(Dense(1936,4536,init='uniform',activation='relu'), name='denseall1',input='flatten96')
graph.add_node(Dense(5808,6776,init='uniform'), name='denseall2',input='flatten96')
######Compute Loss
graph.add_output(name='outputall', input='denseall2')
print "yes"
######Parameters

graph.compile(optimizer=sgd,loss={'outputall':'mse'})
#graph.load_weights('/home/zxy_wqf/cnnold/graphmodeloldexp.h5')
"""
cnt=-1

seq=Sequential()
seq.add(Convolution2D(8,1,3,3,activation='relu'))
seq.add(Convolution2D(12,8,3,3,activation='relu'))
seq.add(MaxPooling2D(poolsize=(2,2)))
seq.add(Convolution2D(14,12,3,3,activation='relu'))
seq.add(Convolution2D(16,14,3,3,activation='relu'))
seq.add(MaxPooling2D(poolsize=(2,2)))
seq.add(Convolution2D(24,16,4,4,activation='relu'))
seq.add(Convolution2D(32,24,3,3,activation='relu'))
seq.add(MaxPooling2D(poolsize=(2,2)))

seq.add(Flatten())

seq.add(Dense(2048,4536,init='uniform',activation='relu'))
seq.add(Dense(4536,4536,init='uniform',activation='relu'))
seq.add(Dense(4536,6776,init='uniform'))
sgd = SGD(lr=0.5, gamma=0.0005, power=0.75, momentum=0.9, nesterov=True)
seq.compile(loss='mse',optimizer=sgd)
seq.load_weights('/home/zxy_wqf/cnnold/seqmodeloldexp.h5')

minloss=0.004776

for epoch in range(0,1):
   for potionid in range(1,2):          #1-6  data batch 12126
      for kind in range(1,2):      #1-5
         cnt=cnt+1   
         print epoch,kind,potionid
         filename="%s%d%s" % ("/home/zxy_wqf/cnnold/trial96_",2,".h5")
         X96=HDF5Matrix(filename,'data96',0,12125)

         print X96.data
         data96 = np.empty(shape=X96.data.shape, dtype="float32")
         data96[:]=X96.data[:]

         #data96=data96[2000:12000]
         #data96=data96.astype("float32")

         Y96=HDF5Matrix(filename,'label96',0,12125)
         label96 = np.empty(shape=Y96.data.shape, dtype="float32")
         label96[:]=Y96.data[:]
         label96=label96[2000:12000]
 
         predictions=seq.predict(data96)
         print predictions
         for id in range(0,12125):
            heatmap=Image.new("L",(22,22))
            a=heatmap.load()
            for i in range(6,7):
               maxx=0.0 
               minx=111   
               for j in range(0,22):
                  for k in range(0,22):
                     minx=min(minx,max(predictions[id][i*484+j*22+k],0))
                     maxx=max(maxx,predictions[id][i*484+j*22+k])      
               for j in range(0,22):
                  for k in range(0,22):
                     a[j,k]=max(a[j,k],(max(predictions[id][i*484+j*22+k],0)-minx)/(maxx-minx)*255.0)
            
               heatmap=heatmap.resize((100,100),Image.ANTIALIAS)             
               filename="/home/zxy_wqf/cnnold/test/"+str(id)+".png"
               heatmap.save(filename)   