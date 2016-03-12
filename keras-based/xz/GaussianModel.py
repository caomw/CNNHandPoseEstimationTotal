import numpy as np
import theano
import theano.tensor as T

from keras.layers.core import Layer

floatX=theano.config.floatX


class GaussianModel(Layer):
       def __init__(self):
       	super(GaussianModel, self).__init__()
       	self.params=[]
       	self.input=T.dmatrix()

       def get_output(self ,train):
       	print train
       	X=self.get_input(train)       	
       	print X
       	num_batch=1
       	num_ball=48
       	height=96
       	width=96
       	d=[]
       	for id in range(0,num_batch):
       	   for i in range(0,height):
       	      for j in range(0,width):
       	         d.append(0.0)
       	         for k in range(0,num_ball):
       	            print id,i,j,k
      	            d[id*height*width+i*width+j]=d[id*height*width+i*width+j] + X[id][3*k+2]*T.exp(-1.0/60.0*( T.pow( (X[id][3*k+0]/X[id][3*k+2]+0.5)*48-i ,2 )  + T.pow( (X[id][3*k+1]/X[id][3*k+2]+0.5)*48-j ,2 ) ))       	
       	output=T.reshape(d,(num_batch,1,height,width))
       	return output