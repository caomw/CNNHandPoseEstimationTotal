import numpy as np
import theano
import theano.tensor as T
import math
from keras.layers.core import Layer
#from seya.utils import apply_model

floatX = theano.config.floatX
class OnlyFlatten(Layer):
    def __init__(self,prenet):
        self.params=[]
        self.prenet=prenet
        self.input=self.prenet.input
    def get_output(self,train=False):
        XX=self.get_input(train)
        X=self.prenet.get_output()
        print 'X',X
        return X


class GaussianModel(Layer):
    def __init__(self,height,width,ballnum,sigma,batchsize):
       super(GaussianModel,self).__init__()
       self.input=T.matrix()
       self.middle=[]
       self.height=height
       self.width=width
       self.ballnum=ballnum
       self.sigma=sigma
       self.batchsize=batchsize

    def get_output(self, train=False):
        #X=self.get_input(train)        
        #outball=apply_model(self.nownet,X)    
        outball=self.get_input(train)
        self.middle = outball
        ##########Main Function
        #Read Data        
        #outmap=[]

        ox1=100.0*self.middle[:,(0)*2].reshape((self.batchsize,1))
        
        x1=T.tile(ox1,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy1=100.0*self.middle[:,(0)*2+1].reshape((self.batchsize,1))
        
        y1=T.tile(oy1,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w1=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h1=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply1=T.tile(self.middle[:,(0)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results1=T.exp((T.minimum(T.pow(x1-w1,2)+T.pow(y1-h1,2),140))*(-1.0/self.sigma))
        #print x1.ndim,y1.ndim,w1.ndim,h1.ndim,multiply1.ndim,results1.ndim

        ox2=100.0*self.middle[:,(1)*2].reshape((self.batchsize,1))
        
        x2=T.tile(ox2,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy2=100.0*self.middle[:,(1)*2+1].reshape((self.batchsize,1))
        
        y2=T.tile(oy2,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w2=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h2=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply2=T.tile(self.middle[:,(1)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results2=T.exp((T.minimum(T.pow(x2-w2,2)+T.pow(y2-h2,2),140))*(-1.0/self.sigma))

        ox3=100.0*self.middle[:,(2)*2].reshape((self.batchsize,1))
        
        x3=T.tile(ox3,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy3=100.0*self.middle[:,(2)*2+1].reshape((self.batchsize,1))
        
        y3=T.tile(oy3,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w3=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h3=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply3=T.tile(self.middle[:,(2)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results3=T.exp((T.minimum(T.pow(x3-w3,2)+T.pow(y3-h3,2),140))*(-1.0/self.sigma))

        ox4=100.0*self.middle[:,(3)*2].reshape((self.batchsize,1))
        
        x4=T.tile(ox4,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy4=100.0*self.middle[:,(3)*2+1].reshape((self.batchsize,1))
        
        y4=T.tile(oy4,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w4=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h4=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply4=T.tile(self.middle[:,(3)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results4=T.exp((T.minimum(T.pow(x4-w4,2)+T.pow(y4-h4,2),140))*(-1.0/self.sigma))

        ox5=100.0*self.middle[:,(4)*2].reshape((self.batchsize,1))
        
        x5=T.tile(ox5,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy5=100.0*self.middle[:,(4)*2+1].reshape((self.batchsize,1))
        
        y5=T.tile(oy5,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w5=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h5=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply5=T.tile(self.middle[:,(4)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results5=T.exp((T.minimum(T.pow(x5-w5,2)+T.pow(y5-h5,2),140))*(-1.0/self.sigma))

        ox6=100.0*self.middle[:,(5)*2].reshape((self.batchsize,1))
        
        x6=T.tile(ox6,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy6=100.0*self.middle[:,(5)*2+1].reshape((self.batchsize,1))
        
        y6=T.tile(oy6,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w6=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h6=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply6=T.tile(self.middle[:,(5)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results6=T.exp((T.minimum(T.pow(x6-w6,2)+T.pow(y6-h6,2),140))*(-1.0/self.sigma))

        ox7=100.0*self.middle[:,(6)*2].reshape((self.batchsize,1))
        
        x7=T.tile(ox7,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy7=100.0*self.middle[:,(6)*2+1].reshape((self.batchsize,1))
        
        y7=T.tile(oy7,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w7=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h7=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply7=T.tile(self.middle[:,(6)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results7=T.exp((T.minimum(T.pow(x7-w7,2)+T.pow(y7-h7,2),140))*(-1.0/self.sigma))

        ox8=100.0*self.middle[:,(7)*2].reshape((self.batchsize,1))
        
        x8=T.tile(ox8,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy8=100.0*self.middle[:,(7)*2+1].reshape((self.batchsize,1))
        
        y8=T.tile(oy8,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w8=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h8=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply8=T.tile(self.middle[:,(7)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results8=T.exp((T.minimum(T.pow(x8-w8,2)+T.pow(y8-h8,2),140))*(-1.0/self.sigma))

        ox9=100.0*self.middle[:,(8)*2].reshape((self.batchsize,1))
        
        x9=T.tile(ox9,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy9=100.0*self.middle[:,(8)*2+1].reshape((self.batchsize,1))
        
        y9=T.tile(oy9,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w9=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h9=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply9=T.tile(self.middle[:,(8)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results9=T.exp((T.minimum(T.pow(x9-w9,2)+T.pow(y9-h9,2),140))*(-1.0/self.sigma))

        ox10=100.0*self.middle[:,(9)*2].reshape((self.batchsize,1))
        
        x10=T.tile(ox10,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy10=100.0*self.middle[:,(9)*2+1].reshape((self.batchsize,1))
        
        y10=T.tile(oy10,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w10=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h10=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply10=T.tile(self.middle[:,(9)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results10=T.exp((T.minimum(T.pow(x10-w10,2)+T.pow(y10-h10,2),140))*(-1.0/self.sigma))

        ox11=100.0*self.middle[:,(10)*2].reshape((self.batchsize,1))
        
        x11=T.tile(ox11,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy11=100.0*self.middle[:,(10)*2+1].reshape((self.batchsize,1))
        
        y11=T.tile(oy11,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w11=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h11=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply11=T.tile(self.middle[:,(10)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results11=T.exp((T.minimum(T.pow(x11-w11,2)+T.pow(y11-h11,2),140))*(-1.0/self.sigma))

        ox12=100.0*self.middle[:,(11)*2].reshape((self.batchsize,1))
        
        x12=T.tile(ox12,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy12=100.0*self.middle[:,(11)*2+1].reshape((self.batchsize,1))
        
        y12=T.tile(oy12,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w12=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h12=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply12=T.tile(self.middle[:,(11)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results12=T.exp((T.minimum(T.pow(x12-w12,2)+T.pow(y12-h12,2),140))*(-1.0/self.sigma))

        ox13=100.0*self.middle[:,(12)*2].reshape((self.batchsize,1))
        
        x13=T.tile(ox13,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy13=100.0*self.middle[:,(12)*2+1].reshape((self.batchsize,1))
        
        y13=T.tile(oy13,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w13=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h13=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply13=T.tile(self.middle[:,(12)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results13=T.exp((T.minimum(T.pow(x13-w13,2)+T.pow(y13-h13,2),140))*(-1.0/self.sigma))

        ox14=100.0*self.middle[:,(13)*2].reshape((self.batchsize,1))
        
        x14=T.tile(ox14,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy14=100.0*self.middle[:,(13)*2+1].reshape((self.batchsize,1))
        
        y14=T.tile(oy14,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w14=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h14=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply14=T.tile(self.middle[:,(13)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results14=T.exp((T.minimum(T.pow(x14-w14,2)+T.pow(y14-h14,2),140))*(-1.0/self.sigma))

        ox15=100.0*self.middle[:,(14)*2].reshape((self.batchsize,1))
        
        x15=T.tile(ox15,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy15=100.0*self.middle[:,(14)*2+1].reshape((self.batchsize,1))
        
        y15=T.tile(oy15,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w15=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h15=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply15=T.tile(self.middle[:,(14)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results15=T.exp((T.minimum(T.pow(x15-w15,2)+T.pow(y15-h15,2),140))*(-1.0/self.sigma))

        ox16=100.0*self.middle[:,(15)*2].reshape((self.batchsize,1))
        
        x16=T.tile(ox16,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy16=100.0*self.middle[:,(15)*2+1].reshape((self.batchsize,1))
        
        y16=T.tile(oy16,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w16=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h16=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply16=T.tile(self.middle[:,(15)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results16=T.exp((T.minimum(T.pow(x16-w16,2)+T.pow(y16-h16,2),140))*(-1.0/self.sigma))

        ox17=100.0*self.middle[:,(16)*2].reshape((self.batchsize,1))
        
        x17=T.tile(ox17,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy17=100.0*self.middle[:,(16)*2+1].reshape((self.batchsize,1))
        
        y17=T.tile(oy17,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w17=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h17=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply17=T.tile(self.middle[:,(16)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results17=T.exp((T.minimum(T.pow(x17-w17,2)+T.pow(y17-h17,2),140))*(-1.0/self.sigma))

        ox18=100.0*self.middle[:,(17)*2].reshape((self.batchsize,1))
        
        x18=T.tile(ox18,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy18=100.0*self.middle[:,(17)*2+1].reshape((self.batchsize,1))
        
        y18=T.tile(oy18,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w18=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h18=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply18=T.tile(self.middle[:,(17)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results18=T.exp((T.minimum(T.pow(x18-w18,2)+T.pow(y18-h18,2),140))*(-1.0/self.sigma))

        ox19=100.0*self.middle[:,(18)*2].reshape((self.batchsize,1))
        
        x19=T.tile(ox19,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy19=100.0*self.middle[:,(18)*2+1].reshape((self.batchsize,1))
        
        y19=T.tile(oy19,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w19=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h19=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply19=T.tile(self.middle[:,(18)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results19=T.exp((T.minimum(T.pow(x19-w19,2)+T.pow(y19-h19,2),140))*(-1.0/self.sigma))

        ox20=100.0*self.middle[:,(19)*2].reshape((self.batchsize,1))
        
        x20=T.tile(ox20,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy20=100.0*self.middle[:,(19)*2+1].reshape((self.batchsize,1))
        
        y20=T.tile(oy20,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w20=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h20=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply20=T.tile(self.middle[:,(19)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results20=T.exp((T.minimum(T.pow(x20-w20,2)+T.pow(y20-h20,2),140))*(-1.0/self.sigma))

        ox21=100.0*self.middle[:,(20)*2].reshape((self.batchsize,1))
        
        x21=T.tile(ox21,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy21=100.0*self.middle[:,(20)*2+1].reshape((self.batchsize,1))
        
        y21=T.tile(oy21,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w21=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h21=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply21=T.tile(self.middle[:,(20)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results21=T.exp((T.minimum(T.pow(x21-w21,2)+T.pow(y21-h21,2),140))*(-1.0/self.sigma))

        ox22=100.0*self.middle[:,(21)*2].reshape((self.batchsize,1))
        
        x22=T.tile(ox22,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy22=100.0*self.middle[:,(21)*2+1].reshape((self.batchsize,1))
        y22=T.tile(oy22,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w22=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h22=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply22=T.tile(self.middle[:,(21)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results22=T.exp((T.minimum(T.pow(x22-w22,2)+T.pow(y22-h22,2),140))*(-1.0/self.sigma))

        ox23=100.0*self.middle[:,(22)*2].reshape((self.batchsize,1))
        
        x23=T.tile(ox23,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy23=100.0*self.middle[:,(22)*2+1].reshape((self.batchsize,1))
        
        y23=T.tile(oy23,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w23=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h23=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply23=T.tile(self.middle[:,(22)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results23=T.exp((T.minimum(T.pow(x23-w23,2)+T.pow(y23-h23,2),140))*(-1.0/self.sigma))

        ox24=100.0*self.middle[:,(23)*2].reshape((self.batchsize,1))
        
        x24=T.tile(ox24,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy24=100.0*self.middle[:,(23)*2+1].reshape((self.batchsize,1))
        
        y24=T.tile(oy24,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w24=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h24=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply24=T.tile(self.middle[:,(23)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results24=T.exp((T.minimum(T.pow(x24-w24,2)+T.pow(y24-h24,2),140))*(-1.0/self.sigma))

        ox25=100.0*self.middle[:,(24)*2].reshape((self.batchsize,1))
        
        x25=T.tile(ox25,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy25=100.0*self.middle[:,(24)*2+1].reshape((self.batchsize,1))
        
        y25=T.tile(oy25,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w25=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h25=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply25=T.tile(self.middle[:,(24)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results25=T.exp((T.minimum(T.pow(x25-w25,2)+T.pow(y25-h25,2),140))*(-1.0/self.sigma))

        ox26=100.0*self.middle[:,(25)*2].reshape((self.batchsize,1))
        
        x26=T.tile(ox26,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy26=100.0*self.middle[:,(25)*2+1].reshape((self.batchsize,1))
        
        y26=T.tile(oy26,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w26=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h26=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply26=T.tile(self.middle[:,(25)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results26=T.exp((T.minimum(T.pow(x26-w26,2)+T.pow(y26-h26,2),140))*(-1.0/self.sigma))

        ox27=100.0*self.middle[:,(26)*2].reshape((self.batchsize,1))
        
        x27=T.tile(ox27,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy27=100.0*self.middle[:,(26)*2+1].reshape((self.batchsize,1))
        
        y27=T.tile(oy27,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w27=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h27=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply27=T.tile(self.middle[:,(26)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results27=T.exp((T.minimum(T.pow(x27-w27,2)+T.pow(y27-h27,2),140))*(-1.0/self.sigma))

        ox28=100.0*self.middle[:,(27)*2].reshape((self.batchsize,1))
        
        x28=T.tile(ox28,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy28=100.0*self.middle[:,(27)*2+1].reshape((self.batchsize,1))
        
        y28=T.tile(oy28,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w28=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h28=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply28=T.tile(self.middle[:,(27)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results28=T.exp((T.minimum(T.pow(x28-w28,2)+T.pow(y28-h28,2),140))*(-1.0/self.sigma))

        ox29=100.0*self.middle[:,(28)*2].reshape((self.batchsize,1))
        
        x29=T.tile(ox29,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy29=100.0*self.middle[:,(28)*2+1].reshape((self.batchsize,1))
        y29=T.tile(oy29,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))

        w29=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h29=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply29=T.tile(self.middle[:,(28)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results29=T.exp((T.minimum(T.pow(x29-w29,2)+T.pow(y29-h29,2),140))*(-1.0/self.sigma))

        ox30=100.0*self.middle[:,(29)*2].reshape((self.batchsize,1))
        x30=T.tile(ox30,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy30=100.0*self.middle[:,(29)*2+1].reshape((self.batchsize,1))
        y30=T.tile(oy30,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w30=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h30=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply30=T.tile(self.middle[:,(29)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results30=T.exp((T.minimum(T.pow(x30-w30,2)+T.pow(y30-h30,2),140))*(-1.0/self.sigma))

        ox31=100.0*self.middle[:,(30)*2].reshape((self.batchsize,1))
        x31=T.tile(ox31,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy31=100.0*self.middle[:,(30)*2+1].reshape((self.batchsize,1))
        y31=T.tile(oy31,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w31=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h31=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply31=T.tile(self.middle[:,(30)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results31=T.exp((T.minimum(T.pow(x31-w31,2)+T.pow(y31-h31,2),140))*(-1.0/self.sigma))

        ox32=100.0*self.middle[:,(31)*2].reshape((self.batchsize,1))
        x32=T.tile(ox32,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy32=100.0*self.middle[:,(31)*2+1].reshape((self.batchsize,1))
        y32=T.tile(oy32,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w32=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h32=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply32=T.tile(self.middle[:,(31)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results32=T.exp((T.minimum(T.pow(x32-w32,2)+T.pow(y32-h32,2),140))*(-1.0/self.sigma))

        ox33=100.0*self.middle[:,(32)*2].reshape((self.batchsize,1))
        x33=T.tile(ox33,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy33=100.0*self.middle[:,(32)*2+1].reshape((self.batchsize,1))
        y33=T.tile(oy33,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w33=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h33=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply33=T.tile(self.middle[:,(32)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results33=T.exp((T.minimum(T.pow(x33-w33,2)+T.pow(y33-h33,2),140))*(-1.0/self.sigma))

        ox34=100.0*self.middle[:,(33)*2].reshape((self.batchsize,1))
        x34=T.tile(ox34,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy34=100.0*self.middle[:,(33)*2+1].reshape((self.batchsize,1))
        y34=T.tile(oy34,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w34=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h34=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply34=T.tile(self.middle[:,(33)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results34=T.exp((T.minimum(T.pow(x34-w34,2)+T.pow(y34-h34,2),140))*(-1.0/self.sigma))

        ox35=100.0*self.middle[:,(34)*2].reshape((self.batchsize,1))
        x35=T.tile(ox35,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy35=100.0*self.middle[:,(34)*2+1].reshape((self.batchsize,1))
        y35=T.tile(oy35,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w35=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h35=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply35=T.tile(self.middle[:,(34)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results35=T.exp((T.minimum(T.pow(x35-w35,2)+T.pow(y35-h35,2),140))*(-1.0/self.sigma))

        ox36=100.0*self.middle[:,(35)*2].reshape((self.batchsize,1))
        x36=T.tile(ox36,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy36=100.0*self.middle[:,(35)*2+1].reshape((self.batchsize,1))
        y36=T.tile(oy36,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w36=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h36=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply36=T.tile(self.middle[:,(35)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results36=T.exp((T.minimum(T.pow(x36-w36,2)+T.pow(y36-h36,2),140))*(-1.0/self.sigma))
        """
        ox37=100.0*self.middle[:,(36)*2].reshape((self.batchsize,1))
        x37=T.tile(ox37,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy37=100.0*self.middle[:,(36)*2+1].reshape((self.batchsize,1))
        y37=T.tile(oy37,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w37=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h37=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply37=T.tile(self.middle[:,(36)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results37=T.exp((T.minimum(T.pow(x37-w37,2)+T.pow(y37-h37,2),140))*(-1.0/self.sigma))

        ox38=100.0*self.middle[:,(37)*2].reshape((self.batchsize,1))
        x38=T.tile(ox38,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy38=100.0*self.middle[:,(37)*2+1].reshape((self.batchsize,1))
        y38=T.tile(oy38,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w38=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h38=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply38=T.tile(self.middle[:,(37)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results38=T.exp((T.minimum(T.pow(x38-w38,2)+T.pow(y38-h38,2),140))*(-1.0/self.sigma))

        ox39=100.0*self.middle[:,(38)*2].reshape((self.batchsize,1))
        x39=T.tile(ox39,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy39=100.0*self.middle[:,(38)*2+1].reshape((self.batchsize,1))
        y39=T.tile(oy39,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w39=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h39=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply39=T.tile(self.middle[:,(38)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results39=T.exp((T.minimum(T.pow(x39-w39,2)+T.pow(y39-h39,2),140))*(-1.0/self.sigma))

        ox40=100.0*self.middle[:,(39)*2].reshape((self.batchsize,1))
        x40=T.tile(ox40,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy40=100.0*self.middle[:,(39)*2+1].reshape((self.batchsize,1))
        y40=T.tile(oy40,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w40=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h40=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply40=T.tile(self.middle[:,(39)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results40=T.exp((T.minimum(T.pow(x40-w40,2)+T.pow(y40-h40,2),140))*(-1.0/self.sigma))

        ox41=100.0*self.middle[:,(40)*2].reshape((self.batchsize,1))
        x41=T.tile(ox41,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy41=100.0*self.middle[:,(40)*2+1].reshape((self.batchsize,1))
        y41=T.tile(oy41,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w41=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h41=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply41=T.tile(self.middle[:,(40)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results41=T.exp((T.minimum(T.pow(x41-w41,2)+T.pow(y41-h41,2),140))*(-1.0/self.sigma))

        ox42=100.0*self.middle[:,(41)*2].reshape((self.batchsize,1))
        x42=T.tile(ox42,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy42=100.0*self.middle[:,(41)*2+1].reshape((self.batchsize,1))
        y42=T.tile(oy42,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w42=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h42=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply42=T.tile(self.middle[:,(41)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results42=T.exp((T.minimum(T.pow(x42-w42,2)+T.pow(y42-h42,2),140))*(-1.0/self.sigma))

        ox43=100.0*self.middle[:,(42)*2].reshape((self.batchsize,1))
        x43=T.tile(ox43,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy43=100.0*self.middle[:,(42)*2+1].reshape((self.batchsize,1))
        y43=T.tile(oy43,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w43=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h43=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply43=T.tile(self.middle[:,(42)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results43=T.exp((T.minimum(T.pow(x43-w43,2)+T.pow(y43-h43,2),140))*(-1.0/self.sigma))

        ox44=100.0*self.middle[:,(43)*2].reshape((self.batchsize,1))
        x44=T.tile(ox44,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy44=100.0*self.middle[:,(43)*2+1].reshape((self.batchsize,1))
        y44=T.tile(oy44,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w44=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h44=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply44=T.tile(self.middle[:,(43)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results44=T.exp((T.minimum(T.pow(x44-w44,2)+T.pow(y44-h44,2),140))*(-1.0/self.sigma))

        ox45=100.0*self.middle[:,(44)*2].reshape((self.batchsize,1))
        x45=T.tile(ox45,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy45=100.0*self.middle[:,(44)*2+1].reshape((self.batchsize,1))
        y45=T.tile(oy45,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w45=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h45=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply45=T.tile(self.middle[:,(44)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results45=T.exp((T.minimum(T.pow(x45-w45,2)+T.pow(y45-h45,2),140))*(-1.0/self.sigma))

        ox46=100.0*self.middle[:,(45)*2].reshape((self.batchsize,1))
        x46=T.tile(ox46,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy46=100.0*self.middle[:,(45)*2+1].reshape((self.batchsize,1))
        y46=T.tile(oy46,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w46=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h46=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply46=T.tile(self.middle[:,(45)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results46=T.exp((T.minimum(T.pow(x46-w46,2)+T.pow(y46-h46,2),140))*(-1.0/self.sigma))

        ox47=100.0*self.middle[:,(46)*2].reshape((self.batchsize,1))
        x47=T.tile(ox47,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy47=100.0*self.middle[:,(46)*2+1].reshape((self.batchsize,1))
        y47=T.tile(oy47,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w47=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h47=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply47=T.tile(self.middle[:,(46)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results47=T.exp((T.minimum(T.pow(x47-w47,2)+T.pow(y47-h47,2),140))*(-1.0/self.sigma))

        ox48=100.0*self.middle[:,(47)*2].reshape((self.batchsize,1))
        x48=T.tile(ox48,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy48=100.0*self.middle[:,(47)*2+1].reshape((self.batchsize,1))
        y48=T.tile(oy48,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w48=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h48=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        #multiply48=T.tile(self.middle[:,(47)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results48=T.exp((T.minimum(T.pow(x48-w48,2)+T.pow(y48-h48,2),140))*(-1.0/self.sigma))
        """
        results=theano.shared(np.zeros((self.batchsize,36,96,96),dtype='float64'))
        #results=T.set_subtensor(results[:,48,:,:],0.01)
        results=T.set_subtensor(results[:,0,:,:],results1)
        results=T.set_subtensor(results[:,1,:,:],results2)
        results=T.set_subtensor(results[:,2,:,:],results3)
        results=T.set_subtensor(results[:,3,:,:],results4)
        results=T.set_subtensor(results[:,4,:,:],results5)
        results=T.set_subtensor(results[:,5,:,:],results6)
        results=T.set_subtensor(results[:,6,:,:],results7)
        results=T.set_subtensor(results[:,7,:,:],results8)
        results=T.set_subtensor(results[:,8,:,:],results9)
        results=T.set_subtensor(results[:,9,:,:],results10)
        results=T.set_subtensor(results[:,10,:,:],results11)
        results=T.set_subtensor(results[:,11,:,:],results12)
        results=T.set_subtensor(results[:,12,:,:],results13)
        results=T.set_subtensor(results[:,13,:,:],results14)
        results=T.set_subtensor(results[:,14,:,:],results15)
        results=T.set_subtensor(results[:,15,:,:],results16)
        results=T.set_subtensor(results[:,16,:,:],results17)
        results=T.set_subtensor(results[:,17,:,:],results18)
        results=T.set_subtensor(results[:,18,:,:],results19)
        results=T.set_subtensor(results[:,19,:,:],results20)
        results=T.set_subtensor(results[:,20,:,:],results21)
        results=T.set_subtensor(results[:,21,:,:],results22)
        results=T.set_subtensor(results[:,22,:,:],results23)
        results=T.set_subtensor(results[:,23,:,:],results24)
        results=T.set_subtensor(results[:,24,:,:],results25)
        results=T.set_subtensor(results[:,25,:,:],results26)
        results=T.set_subtensor(results[:,26,:,:],results27)
        results=T.set_subtensor(results[:,27,:,:],results28)
        results=T.set_subtensor(results[:,28,:,:],results29)
        results=T.set_subtensor(results[:,29,:,:],results30)
        results=T.set_subtensor(results[:,30,:,:],results31)
        results=T.set_subtensor(results[:,31,:,:],results32)
        results=T.set_subtensor(results[:,32,:,:],results33)
        results=T.set_subtensor(results[:,33,:,:],results34)
        results=T.set_subtensor(results[:,34,:,:],results35)
        results=T.set_subtensor(results[:,35,:,:],results36)
        """
        results=T.set_subtensor(results[:,36,:,:],results37)
        results=T.set_subtensor(results[:,37,:,:],results38)
        results=T.set_subtensor(results[:,38,:,:],results39)
        results=T.set_subtensor(results[:,39,:,:],results40)
        results=T.set_subtensor(results[:,40,:,:],results41)
        results=T.set_subtensor(results[:,41,:,:],results42)
        results=T.set_subtensor(results[:,42,:,:],results43)
        results=T.set_subtensor(results[:,43,:,:],results44)
        results=T.set_subtensor(results[:,44,:,:],results45)
        results=T.set_subtensor(results[:,45,:,:],results46)
        results=T.set_subtensor(results[:,46,:,:],results47)
        results=T.set_subtensor(results[:,47,:,:],results48)
        """
        results=results.sum(axis=[1])
        minx=T.tile(results.min(axis=[1,2]).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,(self.height*self.width))).reshape((self.batchsize,self.height,self.width))
        maxx=T.tile(results.max(axis=[1,2]).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,(self.height*self.width))).reshape((self.batchsize,self.height,self.width))

        results=(results-minx)/(maxx-minx)
           
        print 'prepare'           
        print 'gen'
        print 'compute'
        
        output=T.reshape(results,(self.batchsize,1,self.height,self.width))
        print output,output.ndim
        return output

 
#####Ball Model

class BallModel(Layer):
    def __init__(self,height,width,batchsize):
        super(BallModel,self).__init__()
        self.input=T.matrix()
        self.height=height
        self.width=width
        self.middle=[]
        #self.prenet=prenet
        #self.input=prenet.input
        self.batchsize=batchsize
    
    def _getrx(self,d):
        r=theano.shared(np.zeros((3,3),dtype='float64'))
        r=T.set_subtensor(r[0,0],1.0)
        r=T.set_subtensor(r[1,1],T.cos(d))
        r=T.set_subtensor(r[1,2],-T.sin(d))
        r=T.set_subtensor(r[2,1],T.sin(d))
        r=T.set_subtensor(r[2,2],T.cos(d))        
        return r

    def _getry(self,d):
        r=theano.shared(np.zeros((3,3),dtype='float64'))
        r=T.set_subtensor(r[1,1],1.0)
        r=T.set_subtensor(r[0,0],T.cos(d))
        r=T.set_subtensor(r[0,2],-T.sin(d))
        r=T.set_subtensor(r[2,0],T.sin(d))
        r=T.set_subtensor(r[2,2],T.cos(d))        
        return r

    def _getrz(self,d):
        r=theano.shared(np.zeros((3,3),dtype='float64'))
        r=T.set_subtensor(r[2,2],1.0)
        r=T.set_subtensor(r[0,0],T.cos(d))
        r=T.set_subtensor(r[0,1],-T.sin(d))
        r=T.set_subtensor(r[1,0],T.sin(d))
        r=T.set_subtensor(r[1,1],T.cos(d))        
        return r
    def _loopoverallbatch(self,batch):
        initial=np.array([[0.27,0.15,0.0],
                [0.09,0.10,0.0],
                [-0.09,0.10,0.0],
                [-0.27,0.10,0.0],
                [0.27,0.30,0.0],
                [0.09,0.30,0.0],
                [-0.09,0.30,0.0],
                [-0.27,0.30,0.0],
                [0.27,0.50,0.0],
                [0.09,0.50,0.0],
                [-0.09,0.50,0.0],
                [-0.27,0.50,0.0],
                [0.28,0.70,0],
                [0.08,0.70,0],
                [-0.11,0.70,0],
                [-0.28,0.70,0],
                [0.41,0.15,0],
                [0.45,0.15,0],
                [0.49,0.15,0],
                [0.53,0.15,0],
                [0.67,0.15,0],
                [0.81,0.15,0],
                [0.94,0.15,0],
                [1.05,0.15,0],
                [0.28,0.88,0],
                [0.28,1.04,0],
                [0.28,1.19,0],
                [0.28,1.33,0],
                [0.28,1.45,0],
                [0.28,1.56,0],
                [0.08,0.90,0],
                [0.08,1.10,0],
                [0.08,1.27,0],
                [0.08,1.41,0],
                [0.08,1.55,0],
                [0.08,1.69,0],
                [-0.11,0.88,0],
                [-0.11,1.04,0],
                [-0.11,1.19,0],
                [-0.11,1.33,0],
                [-0.11,1.45,0],
                [-0.11,1.56,0],
                [-0.28,0.85,0],
                [-0.28,0.95,0],
                [-0.28,1.05,0],
                [-0.28,1.16,0],
                [-0.28,1.27,0],
                [-0.28,1.35,0]])
        d=self.middle[batch].reshape((26,)) ##ndim=0        

        r0=T.dot(T.dot(self._getrx(d[3]),self._getry(d[4])),self._getrz(d[5]))

        x=theano.shared(np.zeros((3,48),dtype='float64'))
        xx=theano.shared(np.zeros((3,36),dtype='float64'))
        for i in range(0,16):
            #x[:,i]=T.dot(r0,initial[i][:].T)+d[0:3]
            x=T.set_subtensor(x[:,i],T.dot(r0,initial[i][:].T)+d[0:3])

        o16=x[:,0]
        r16=T.dot(self._getry(d[6]),self._getrz(d[7]))
        for i in range(16,20):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(r0,r16),(initial[i][:]-initial[0][:]).T)+o16)
        
        """
        o20=x[:,19]
        r20=self._getrz(d[8])
        for i in range(20,22):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(r0,r16),r20),((initial[i][:]-initial[19][:]).T))+o20)
        """
        o22=x[:,21]
        r22=self._getrz(d[9])
        """
        for i in range(22,24):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(T.dot(r0,r16),r20),r22),((initial[i][:]-initial[21][:]).T))+o22)
        """
        for k in range(0,4):
            p=12+k
            o=x[:,p]
            r1=T.dot(self._getrx(d[11+4*k-1]),self._getrz(d[11+4*k]))
            for i in range(25+k*6-1,25+k*6+1):
                x=T.set_subtensor(x[:,i],T.dot(T.dot(r0,r1),((initial[i][:]-initial[p][:]).T))+o)

            p=25+k*6
            o=x[:,p]
            r2=self._getrx(d[13+4*k-1])
            for i in range(p+1,p+3):
                x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(r0,r1),r2),((initial[i][:]-initial[p][:]).T))+o)
            """         
            p=p+2
            o=x[:,p]
            r3=self._getrx(d[14+4*k-1])
            for i in range(p+1,p+3):
                x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(T.dot(r0,r1),r2),r3),((initial[i][:]-initial[p][:]).T))+o)
            """
        
        toadd=np.array([ [-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2],
                         [-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7,-0.7],
                         [ 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]])
        x=x+toadd
        xx=T.set_subtensor(xx[:,0:16],x[:,0:16])
        xx=T.set_subtensor(xx[:,16:20],x[:,16:20])
        xx=T.set_subtensor(xx[:,20:24],x[:,24:28])
        xx=T.set_subtensor(xx[:,24:28],x[:,30:34])
        xx=T.set_subtensor(xx[:,28:32],x[:,36:40])
        xx=T.set_subtensor(xx[:,32:36],x[:,42:46])

        ans=theano.shared(np.zeros((2,36),dtype='float64'))
        ans=T.set_subtensor(ans[0,:],T.maximum(-10,T.minimum((xx[0,:]/T.maximum(xx[2,:],0.1) +0.5)*self.width,105))/100.0)
        ans=T.set_subtensor(ans[1,:],T.maximum(-10,T.minimum((-xx[1,:]/T.maximum(xx[2,:],0.1)+0.5)*self.height,105))/100.0)
        ans=ans.T
        ans=T.reshape(ans,(72,))
        return ans

    def get_output(self, train=False):        
        
        outputans=[]
        X = self.get_input(train)
        self.middle=X

        results, updates = theano.scan(fn=self._loopoverallbatch ,
                                                           sequences=[T.arange(0,self.batchsize)])           

        print results,results.ndim
        return results
      