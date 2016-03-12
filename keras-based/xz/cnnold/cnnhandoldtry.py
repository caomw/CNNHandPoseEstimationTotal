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
sgd = SGD(lr=0.08,sigmma=0.000005, power=0.75, momentum=0.9, nesterov=True)
seq.compile(loss='mse',optimizer=sgd)
seq.load_weights('/home/zxy_wqf/cnnold/seqmodeloldexp.h5')

minloss=0.001307
#minloss=0.002126
x=np.zeros((72756,1,96,96),dtype='float32')
y=np.zeros((72756,6776),dtype='float32')
xx=np.zeros((72756,1,96,96),dtype='float32')
yy=np.zeros((72756,6776),dtype='float32')
ind=np.zeros((72756),dtype='int32')
for i in range(0,72756):
   ind[i]=i
######Load Data
for epoch in range(0,1000):
   for kind in range(1,2):      #1-5
      for potionid in range(1,7):          #1-6  data batch 12126      
         cnt=cnt+1 	
         print epoch,kind,potionid
         filename="%s%d%s" % ("/home/zxy_wqf/cnnold/trial96_",(kind-1)*6+potionid,".h5")
         X96=HDF5Matrix(filename,'data96',0,12125)        
         data96 = np.empty(shape=X96.data.shape, dtype="float32")
         data96[:]=X96.data[:]

         Y96=HDF5Matrix(filename,'label96',0,12125)
         label96 = np.empty(shape=Y96.data.shape, dtype="float32")
         label96[:]=Y96.data[:]

         x[12126*(potionid-1):12126*potionid]=data96[:]
         y[12126*(potionid-1):12126*potionid]=label96[:]
      np.random.shuffle(ind)
      for i in range(0,72756):
         xx[i]=x[ind[i]]
         yy[i]=y[ind[i]]
      seq.fit(xx,yy,batch_size=16,nb_epoch=1,shuffle=True)
      loss=seq.evaluate(xx,yy) 
      print 'nowloss:',loss
      if loss<minloss:
         minloss=loss            
         seq.save_weights('/home/zxy_wqf/cnnold/seqmodeloldexp.h5',overwrite=True)
      print 'minloss:', minloss
         
      """
         for id in range(0,14):
            for row in range(0,22):
               for col in range(0,22):
                  print "%.1f " %(label96[13][id*484+row*22+col]) ,
               print '\n'
            print '--------------------------------------------\n'
          
         graph.fit({'oinput96':data96,'outputall': label96}, batch_size=32,nb_epoch=1,shuffle=True)         
         
         seq.fit(data96,label96,batch_size=32,nb_epoch=1,shuffle=True)
         
         evalualoss=0.0
         

         filename="%s%d%s" % ("/home/zxy_wqf/cnnold/trial96_",(kind-1)*6+potionid,".h5")
         X96=HDF5Matrix(filename,'data96',0,12125)
            
         data96 = np.empty(shape=X96.data.shape, dtype="float32")
         data96[:]=X96.data[:]
         data96=data96.astype("float32")            

         Y96=HDF5Matrix(filename,'label96',0,12125)
         label96 = np.empty(shape=Y96.data.shape, dtype="float32")
         label96[:]=Y96.data[:]
         label96=label96.astype("float32")
            
         loss=seq.evaluate(data96,label96) 
         print loss
         if potionid==1 and loss<minloss:
            minloss=loss            
            seq.save_weights('/home/zxy_wqf/cnnold/seqmodeloldexp.h5',overwrite=True)
         print minloss
      """
#predictions=graph.predict({'oinput96':data96,'oinput48':data48,'oinput24':data24})
#heatmap=Image.new("L",(22,22))
#a=heatmap.load()
#sum=0.0

#minx=1111.0
#maxx=0.0
#for id in range(359,360):
   #for  i in range(0,14):
      #for j in range(0,22):
         #for k in range(0,22):
            #minx=min(minx,predictions.values()[0][id][i*484+j*22+k])	
            #maxx=max(maxx,predictions.values()[0][id][i*484+j*22+k])
#for id in range(359,360):
   #for  i in range(0,14):
      #for j in range(0,22):
         #for k in range(0,22):
             #a[j,k]=(predictions.values()[0][id][i*484+j*22+k]-minx)/(maxx-minx)*255.0
#print minx,maxx
#heatmap=heatmap.resize((500,500),Image.ANTIALIAS)             
            #sum=sum+math.fabs(predictions.values()[0][id][i*484+j*22+k]-label96[id,i*484+j*22+k])
#sum=sum/6776.0/1000.0
#print sum
#heatmap.show()