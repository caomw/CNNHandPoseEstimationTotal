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
        outmap=[]
        a=np.zeros((self.ballnum),dtype="int32")
        b=np.zeros((self.ballnum),dtype="int32")                        
        for batchid in range(0,self.batchsize):
           #for i in range(0,self.ballnum):
              #a[i]=i
              #b[i]=batch
           #ballshare=theano.shared(np.array(a).T)           
           #batchshare=theano.shared(np.array(b).T)
           #results, updates = theano.scan(fn=self._loopoverallball ,
                                                           #sequences=[ballshare,batchshare])           
           ox1=self.middle[batchid][(0)*3].reshape((1,1))
           x1=T.tile(ox1,(self.height,self.width))
           oy1=self.middle[batchid][(0)*3+1].reshape((1,1))
           y1=T.tile(oy1,(self.height,self.width))
           w1=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h1=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results1=self.middle[batchid][(0)*3+2]*T.exp((T.pow(x1-w1,2)+T.pow(y1-h1,2))*(-1.0/self.sigma))

           ox2=self.middle[batchid][(1)*3].reshape((1,1))
           x2=T.tile(ox2,(self.height,self.width))
           oy2=self.middle[batchid][(1)*3+1].reshape((1,1))
           y2=T.tile(oy2,(self.height,self.width))
           w2=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h2=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results2=self.middle[batchid][(1)*3+2]*T.exp((T.pow(x2-w2,2)+T.pow(y2-h2,2))*(-1.0/self.sigma))

           ox3=self.middle[batchid][(2)*3].reshape((1,1))
           x3=T.tile(ox3,(self.height,self.width))
           oy3=self.middle[batchid][(2)*3+1].reshape((1,1))
           y3=T.tile(oy3,(self.height,self.width))
           w3=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h3=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results3=self.middle[batchid][(2)*3+2]*T.exp((T.pow(x3-w3,2)+T.pow(y3-h3,2))*(-1.0/self.sigma))

           ox4=self.middle[batchid][(3)*3].reshape((1,1))
           x4=T.tile(ox4,(self.height,self.width))
           oy4=self.middle[batchid][(3)*3+1].reshape((1,1))
           y4=T.tile(oy4,(self.height,self.width))
           w4=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h4=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results4=self.middle[batchid][(3)*3+2]*T.exp((T.pow(x4-w4,2)+T.pow(y4-h4,2))*(-1.0/self.sigma))

           ox5=self.middle[batchid][(4)*3].reshape((1,1))
           x5=T.tile(ox5,(self.height,self.width))
           oy5=self.middle[batchid][(4)*3+1].reshape((1,1))
           y5=T.tile(oy5,(self.height,self.width))
           w5=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h5=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results5=self.middle[batchid][(4)*3+2]*T.exp((T.pow(x5-w5,2)+T.pow(y5-h5,2))*(-1.0/self.sigma))

           ox6=self.middle[batchid][(5)*3].reshape((1,1))
           x6=T.tile(ox6,(self.height,self.width))
           oy6=self.middle[batchid][(5)*3+1].reshape((1,1))
           y6=T.tile(oy6,(self.height,self.width))
           w6=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h6=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results6=self.middle[batchid][(5)*3+2]*T.exp((T.pow(x6-w6,2)+T.pow(y6-h6,2))*(-1.0/self.sigma))

           ox7=self.middle[batchid][(6)*3].reshape((1,1))
           x7=T.tile(ox7,(self.height,self.width))
           oy7=self.middle[batchid][(6)*3+1].reshape((1,1))
           y7=T.tile(oy7,(self.height,self.width))
           w7=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h7=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results7=self.middle[batchid][(6)*3+2]*T.exp((T.pow(x7-w7,2)+T.pow(y7-h7,2))*(-1.0/self.sigma))

           ox8=self.middle[batchid][(7)*3].reshape((1,1))
           x8=T.tile(ox8,(self.height,self.width))
           oy8=self.middle[batchid][(7)*3+1].reshape((1,1))
           y8=T.tile(oy8,(self.height,self.width))
           w8=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h8=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results8=self.middle[batchid][(7)*3+2]*T.exp((T.pow(x8-w8,2)+T.pow(y8-h8,2))*(-1.0/self.sigma))

           ox9=self.middle[batchid][(8)*3].reshape((1,1))
           x9=T.tile(ox9,(self.height,self.width))
           oy9=self.middle[batchid][(8)*3+1].reshape((1,1))
           y9=T.tile(oy9,(self.height,self.width))
           w9=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h9=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results9=self.middle[batchid][(8)*3+2]*T.exp((T.pow(x9-w9,2)+T.pow(y9-h9,2))*(-1.0/self.sigma))

           ox10=self.middle[batchid][(9)*3].reshape((1,1))
           x10=T.tile(ox10,(self.height,self.width))
           oy10=self.middle[batchid][(9)*3+1].reshape((1,1))
           y10=T.tile(oy10,(self.height,self.width))
           w10=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h10=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results10=self.middle[batchid][(9)*3+2]*T.exp((T.pow(x10-w10,2)+T.pow(y10-h10,2))*(-1.0/self.sigma))

           ox11=self.middle[batchid][(10)*3].reshape((1,1))
           x11=T.tile(ox11,(self.height,self.width))
           oy11=self.middle[batchid][(10)*3+1].reshape((1,1))
           y11=T.tile(oy11,(self.height,self.width))
           w11=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h11=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results11=self.middle[batchid][(10)*3+2]*T.exp((T.pow(x11-w11,2)+T.pow(y11-h11,2))*(-1.0/self.sigma))

           ox12=self.middle[batchid][(11)*3].reshape((1,1))
           x12=T.tile(ox12,(self.height,self.width))
           oy12=self.middle[batchid][(11)*3+1].reshape((1,1))
           y12=T.tile(oy12,(self.height,self.width))
           w12=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h12=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results12=self.middle[batchid][(11)*3+2]*T.exp((T.pow(x12-w12,2)+T.pow(y12-h12,2))*(-1.0/self.sigma))

           ox13=self.middle[batchid][(12)*3].reshape((1,1))
           x13=T.tile(ox13,(self.height,self.width))
           oy13=self.middle[batchid][(12)*3+1].reshape((1,1))
           y13=T.tile(oy13,(self.height,self.width))
           w13=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h13=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results13=self.middle[batchid][(12)*3+2]*T.exp((T.pow(x13-w13,2)+T.pow(y13-h13,2))*(-1.0/self.sigma))

           ox14=self.middle[batchid][(13)*3].reshape((1,1))
           x14=T.tile(ox14,(self.height,self.width))
           oy14=self.middle[batchid][(13)*3+1].reshape((1,1))
           y14=T.tile(oy14,(self.height,self.width))
           w14=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h14=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results14=self.middle[batchid][(13)*3+2]*T.exp((T.pow(x14-w14,2)+T.pow(y14-h14,2))*(-1.0/self.sigma))

           ox15=self.middle[batchid][(14)*3].reshape((1,1))
           x15=T.tile(ox15,(self.height,self.width))
           oy15=self.middle[batchid][(14)*3+1].reshape((1,1))
           y15=T.tile(oy15,(self.height,self.width))
           w15=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h15=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results15=self.middle[batchid][(14)*3+2]*T.exp((T.pow(x15-w15,2)+T.pow(y15-h15,2))*(-1.0/self.sigma))

           ox16=self.middle[batchid][(15)*3].reshape((1,1))
           x16=T.tile(ox16,(self.height,self.width))
           oy16=self.middle[batchid][(15)*3+1].reshape((1,1))
           y16=T.tile(oy16,(self.height,self.width))
           w16=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h16=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results16=self.middle[batchid][(15)*3+2]*T.exp((T.pow(x16-w16,2)+T.pow(y16-h16,2))*(-1.0/self.sigma))

           ox17=self.middle[batchid][(16)*3].reshape((1,1))
           x17=T.tile(ox17,(self.height,self.width))
           oy17=self.middle[batchid][(16)*3+1].reshape((1,1))
           y17=T.tile(oy17,(self.height,self.width))
           w17=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h17=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results17=self.middle[batchid][(16)*3+2]*T.exp((T.pow(x17-w17,2)+T.pow(y17-h17,2))*(-1.0/self.sigma))

           ox18=self.middle[batchid][(17)*3].reshape((1,1))
           x18=T.tile(ox18,(self.height,self.width))
           oy18=self.middle[batchid][(17)*3+1].reshape((1,1))
           y18=T.tile(oy18,(self.height,self.width))
           w18=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h18=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results18=self.middle[batchid][(17)*3+2]*T.exp((T.pow(x18-w18,2)+T.pow(y18-h18,2))*(-1.0/self.sigma))

           ox19=self.middle[batchid][(18)*3].reshape((1,1))
           x19=T.tile(ox19,(self.height,self.width))
           oy19=self.middle[batchid][(18)*3+1].reshape((1,1))
           y19=T.tile(oy19,(self.height,self.width))
           w19=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h19=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results19=self.middle[batchid][(18)*3+2]*T.exp((T.pow(x19-w19,2)+T.pow(y19-h19,2))*(-1.0/self.sigma))

           ox20=self.middle[batchid][(19)*3].reshape((1,1))
           x20=T.tile(ox20,(self.height,self.width))
           oy20=self.middle[batchid][(19)*3+1].reshape((1,1))
           y20=T.tile(oy20,(self.height,self.width))
           w20=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h20=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results20=self.middle[batchid][(19)*3+2]*T.exp((T.pow(x20-w20,2)+T.pow(y20-h20,2))*(-1.0/self.sigma))

           ox21=self.middle[batchid][(20)*3].reshape((1,1))
           x21=T.tile(ox21,(self.height,self.width))
           oy21=self.middle[batchid][(20)*3+1].reshape((1,1))
           y21=T.tile(oy21,(self.height,self.width))
           w21=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h21=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results21=self.middle[batchid][(20)*3+2]*T.exp((T.pow(x21-w21,2)+T.pow(y21-h21,2))*(-1.0/self.sigma))

           ox22=self.middle[batchid][(21)*3].reshape((1,1))
           x22=T.tile(ox22,(self.height,self.width))
           oy22=self.middle[batchid][(21)*3+1].reshape((1,1))
           y22=T.tile(oy22,(self.height,self.width))
           w22=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h22=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results22=self.middle[batchid][(21)*3+2]*T.exp((T.pow(x22-w22,2)+T.pow(y22-h22,2))*(-1.0/self.sigma))

           ox23=self.middle[batchid][(22)*3].reshape((1,1))
           x23=T.tile(ox23,(self.height,self.width))
           oy23=self.middle[batchid][(22)*3+1].reshape((1,1))
           y23=T.tile(oy23,(self.height,self.width))
           w23=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h23=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results23=self.middle[batchid][(22)*3+2]*T.exp((T.pow(x23-w23,2)+T.pow(y23-h23,2))*(-1.0/self.sigma))

           ox24=self.middle[batchid][(23)*3].reshape((1,1))
           x24=T.tile(ox24,(self.height,self.width))
           oy24=self.middle[batchid][(23)*3+1].reshape((1,1))
           y24=T.tile(oy24,(self.height,self.width))
           w24=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h24=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results24=self.middle[batchid][(23)*3+2]*T.exp((T.pow(x24-w24,2)+T.pow(y24-h24,2))*(-1.0/self.sigma))

           ox25=self.middle[batchid][(24)*3].reshape((1,1))
           x25=T.tile(ox25,(self.height,self.width))
           oy25=self.middle[batchid][(24)*3+1].reshape((1,1))
           y25=T.tile(oy25,(self.height,self.width))
           w25=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h25=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results25=self.middle[batchid][(24)*3+2]*T.exp((T.pow(x25-w25,2)+T.pow(y25-h25,2))*(-1.0/self.sigma))

           ox26=self.middle[batchid][(25)*3].reshape((1,1))
           x26=T.tile(ox26,(self.height,self.width))
           oy26=self.middle[batchid][(25)*3+1].reshape((1,1))
           y26=T.tile(oy26,(self.height,self.width))
           w26=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h26=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results26=self.middle[batchid][(25)*3+2]*T.exp((T.pow(x26-w26,2)+T.pow(y26-h26,2))*(-1.0/self.sigma))

           ox27=self.middle[batchid][(26)*3].reshape((1,1))
           x27=T.tile(ox27,(self.height,self.width))
           oy27=self.middle[batchid][(26)*3+1].reshape((1,1))
           y27=T.tile(oy27,(self.height,self.width))
           w27=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h27=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results27=self.middle[batchid][(26)*3+2]*T.exp((T.pow(x27-w27,2)+T.pow(y27-h27,2))*(-1.0/self.sigma))

           ox28=self.middle[batchid][(27)*3].reshape((1,1))
           x28=T.tile(ox28,(self.height,self.width))
           oy28=self.middle[batchid][(27)*3+1].reshape((1,1))
           y28=T.tile(oy28,(self.height,self.width))
           w28=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h28=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results28=self.middle[batchid][(27)*3+2]*T.exp((T.pow(x28-w28,2)+T.pow(y28-h28,2))*(-1.0/self.sigma))

           ox29=self.middle[batchid][(28)*3].reshape((1,1))
           x29=T.tile(ox29,(self.height,self.width))
           oy29=self.middle[batchid][(28)*3+1].reshape((1,1))
           y29=T.tile(oy29,(self.height,self.width))
           w29=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h29=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results29=self.middle[batchid][(28)*3+2]*T.exp((T.pow(x29-w29,2)+T.pow(y29-h29,2))*(-1.0/self.sigma))

           ox30=self.middle[batchid][(29)*3].reshape((1,1))
           x30=T.tile(ox30,(self.height,self.width))
           oy30=self.middle[batchid][(29)*3+1].reshape((1,1))
           y30=T.tile(oy30,(self.height,self.width))
           w30=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h30=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results30=self.middle[batchid][(29)*3+2]*T.exp((T.pow(x30-w30,2)+T.pow(y30-h30,2))*(-1.0/self.sigma))

           ox31=self.middle[batchid][(30)*3].reshape((1,1))
           x31=T.tile(ox31,(self.height,self.width))
           oy31=self.middle[batchid][(30)*3+1].reshape((1,1))
           y31=T.tile(oy31,(self.height,self.width))
           w31=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h31=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results31=self.middle[batchid][(30)*3+2]*T.exp((T.pow(x31-w31,2)+T.pow(y31-h31,2))*(-1.0/self.sigma))

           ox32=self.middle[batchid][(31)*3].reshape((1,1))
           x32=T.tile(ox32,(self.height,self.width))
           oy32=self.middle[batchid][(31)*3+1].reshape((1,1))
           y32=T.tile(oy32,(self.height,self.width))
           w32=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h32=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results32=self.middle[batchid][(31)*3+2]*T.exp((T.pow(x32-w32,2)+T.pow(y32-h32,2))*(-1.0/self.sigma))

           ox33=self.middle[batchid][(32)*3].reshape((1,1))
           x33=T.tile(ox33,(self.height,self.width))
           oy33=self.middle[batchid][(32)*3+1].reshape((1,1))
           y33=T.tile(oy33,(self.height,self.width))
           w33=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h33=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results33=self.middle[batchid][(32)*3+2]*T.exp((T.pow(x33-w33,2)+T.pow(y33-h33,2))*(-1.0/self.sigma))

           ox34=self.middle[batchid][(33)*3].reshape((1,1))
           x34=T.tile(ox34,(self.height,self.width))
           oy34=self.middle[batchid][(33)*3+1].reshape((1,1))
           y34=T.tile(oy34,(self.height,self.width))
           w34=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h34=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results34=self.middle[batchid][(33)*3+2]*T.exp((T.pow(x34-w34,2)+T.pow(y34-h34,2))*(-1.0/self.sigma))

           ox35=self.middle[batchid][(34)*3].reshape((1,1))
           x35=T.tile(ox35,(self.height,self.width))
           oy35=self.middle[batchid][(34)*3+1].reshape((1,1))
           y35=T.tile(oy35,(self.height,self.width))
           w35=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h35=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results35=self.middle[batchid][(34)*3+2]*T.exp((T.pow(x35-w35,2)+T.pow(y35-h35,2))*(-1.0/self.sigma))

           ox36=self.middle[batchid][(35)*3].reshape((1,1))
           x36=T.tile(ox36,(self.height,self.width))
           oy36=self.middle[batchid][(35)*3+1].reshape((1,1))
           y36=T.tile(oy36,(self.height,self.width))
           w36=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h36=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results36=self.middle[batchid][(35)*3+2]*T.exp((T.pow(x36-w36,2)+T.pow(y36-h36,2))*(-1.0/self.sigma))

           ox37=self.middle[batchid][(36)*3].reshape((1,1))
           x37=T.tile(ox37,(self.height,self.width))
           oy37=self.middle[batchid][(36)*3+1].reshape((1,1))
           y37=T.tile(oy37,(self.height,self.width))
           w37=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h37=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results37=self.middle[batchid][(36)*3+2]*T.exp((T.pow(x37-w37,2)+T.pow(y37-h37,2))*(-1.0/self.sigma))

           ox38=self.middle[batchid][(37)*3].reshape((1,1))
           x38=T.tile(ox38,(self.height,self.width))
           oy38=self.middle[batchid][(37)*3+1].reshape((1,1))
           y38=T.tile(oy38,(self.height,self.width))
           w38=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h38=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results38=self.middle[batchid][(37)*3+2]*T.exp((T.pow(x38-w38,2)+T.pow(y38-h38,2))*(-1.0/self.sigma))

           ox39=self.middle[batchid][(38)*3].reshape((1,1))
           x39=T.tile(ox39,(self.height,self.width))
           oy39=self.middle[batchid][(38)*3+1].reshape((1,1))
           y39=T.tile(oy39,(self.height,self.width))
           w39=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h39=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results39=self.middle[batchid][(38)*3+2]*T.exp((T.pow(x39-w39,2)+T.pow(y39-h39,2))*(-1.0/self.sigma))

           ox40=self.middle[batchid][(39)*3].reshape((1,1))
           x40=T.tile(ox40,(self.height,self.width))
           oy40=self.middle[batchid][(39)*3+1].reshape((1,1))
           y40=T.tile(oy40,(self.height,self.width))
           w40=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h40=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results40=self.middle[batchid][(39)*3+2]*T.exp((T.pow(x40-w40,2)+T.pow(y40-h40,2))*(-1.0/self.sigma))

           ox41=self.middle[batchid][(40)*3].reshape((1,1))
           x41=T.tile(ox41,(self.height,self.width))
           oy41=self.middle[batchid][(40)*3+1].reshape((1,1))
           y41=T.tile(oy41,(self.height,self.width))
           w41=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h41=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results41=self.middle[batchid][(40)*3+2]*T.exp((T.pow(x41-w41,2)+T.pow(y41-h41,2))*(-1.0/self.sigma))

           ox42=self.middle[batchid][(41)*3].reshape((1,1))
           x42=T.tile(ox42,(self.height,self.width))
           oy42=self.middle[batchid][(41)*3+1].reshape((1,1))
           y42=T.tile(oy42,(self.height,self.width))
           w42=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h42=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results42=self.middle[batchid][(41)*3+2]*T.exp((T.pow(x42-w42,2)+T.pow(y42-h42,2))*(-1.0/self.sigma))

           ox43=self.middle[batchid][(42)*3].reshape((1,1))
           x43=T.tile(ox43,(self.height,self.width))
           oy43=self.middle[batchid][(42)*3+1].reshape((1,1))
           y43=T.tile(oy43,(self.height,self.width))
           w43=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h43=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results43=self.middle[batchid][(42)*3+2]*T.exp((T.pow(x43-w43,2)+T.pow(y43-h43,2))*(-1.0/self.sigma))

           ox44=self.middle[batchid][(43)*3].reshape((1,1))
           x44=T.tile(ox44,(self.height,self.width))
           oy44=self.middle[batchid][(43)*3+1].reshape((1,1))
           y44=T.tile(oy44,(self.height,self.width))
           w44=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h44=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results44=self.middle[batchid][(43)*3+2]*T.exp((T.pow(x44-w44,2)+T.pow(y44-h44,2))*(-1.0/self.sigma))

           ox45=self.middle[batchid][(44)*3].reshape((1,1))
           x45=T.tile(ox45,(self.height,self.width))
           oy45=self.middle[batchid][(44)*3+1].reshape((1,1))
           y45=T.tile(oy45,(self.height,self.width))
           w45=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h45=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results45=self.middle[batchid][(44)*3+2]*T.exp((T.pow(x45-w45,2)+T.pow(y45-h45,2))*(-1.0/self.sigma))

           ox46=self.middle[batchid][(45)*3].reshape((1,1))
           x46=T.tile(ox46,(self.height,self.width))
           oy46=self.middle[batchid][(45)*3+1].reshape((1,1))
           y46=T.tile(oy46,(self.height,self.width))
           w46=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h46=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results46=self.middle[batchid][(45)*3+2]*T.exp((T.pow(x46-w46,2)+T.pow(y46-h46,2))*(-1.0/self.sigma))

           ox47=self.middle[batchid][(46)*3].reshape((1,1))
           x47=T.tile(ox47,(self.height,self.width))
           oy47=self.middle[batchid][(46)*3+1].reshape((1,1))
           y47=T.tile(oy47,(self.height,self.width))
           w47=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h47=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results47=self.middle[batchid][(46)*3+2]*T.exp((T.pow(x47-w47,2)+T.pow(y47-h47,2))*(-1.0/self.sigma))

           ox48=self.middle[batchid][(47)*3].reshape((1,1))
           x48=T.tile(ox48,(self.height,self.width))
           oy48=self.middle[batchid][(47)*3+1].reshape((1,1))
           y48=T.tile(oy48,(self.height,self.width))
           w48=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
           h48=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
           results48=self.middle[batchid][(47)*3+2]*T.exp((T.pow(x48-w48,2)+T.pow(y48-h48,2))*(-1.0/self.sigma))


           ###Set Result
           results=theano.shared(np.zeros((48,96,96),dtype='float32'))
           results=T.set_subtensor(results[0],results1)
           results=T.set_subtensor(results[1],results2)
           results=T.set_subtensor(results[2],results3)
           results=T.set_subtensor(results[3],results4)
           results=T.set_subtensor(results[4],results5)
           results=T.set_subtensor(results[5],results6)
           results=T.set_subtensor(results[6],results7)
           results=T.set_subtensor(results[7],results8)
           results=T.set_subtensor(results[8],results9)
           results=T.set_subtensor(results[9],results10)
           results=T.set_subtensor(results[10],results11)
           results=T.set_subtensor(results[11],results12)
           results=T.set_subtensor(results[12],results13)
           results=T.set_subtensor(results[13],results14)
           results=T.set_subtensor(results[14],results15)
           results=T.set_subtensor(results[15],results16)
           results=T.set_subtensor(results[16],results17)
           results=T.set_subtensor(results[17],results18)
           results=T.set_subtensor(results[18],results19)
           results=T.set_subtensor(results[19],results20)
           results=T.set_subtensor(results[20],results21)
           results=T.set_subtensor(results[21],results22)
           results=T.set_subtensor(results[22],results23)
           results=T.set_subtensor(results[23],results24)
           results=T.set_subtensor(results[24],results25)
           results=T.set_subtensor(results[25],results26)
           results=T.set_subtensor(results[26],results27)
           results=T.set_subtensor(results[27],results28)
           results=T.set_subtensor(results[28],results29)
           results=T.set_subtensor(results[29],results30)
           results=T.set_subtensor(results[30],results31)
           results=T.set_subtensor(results[31],results32)
           results=T.set_subtensor(results[32],results33)
           results=T.set_subtensor(results[33],results34)
           results=T.set_subtensor(results[34],results35)
           results=T.set_subtensor(results[35],results36)
           results=T.set_subtensor(results[36],results37)
           results=T.set_subtensor(results[37],results38)
           results=T.set_subtensor(results[38],results39)
           results=T.set_subtensor(results[39],results40)
           results=T.set_subtensor(results[40],results41)
           results=T.set_subtensor(results[41],results42)
           results=T.set_subtensor(results[42],results43)
           results=T.set_subtensor(results[43],results44)
           results=T.set_subtensor(results[44],results45)
           results=T.set_subtensor(results[45],results46)
           results=T.set_subtensor(results[46],results47)
           results=T.set_subtensor(results[47],results48)           
           results=results.sum(axis=[0])
           fmin=T.tile(results.min().reshape((1,1)),(96,96))
           maxx=results.max()
           minx=results.min()
           results=(results-fmin)/(maxx-minx)
           print results1.ndim,results.ndim
           #print 'dim1',results.ndim
           
           #print 'dim2',results.ndim
           #print results
           print 'prepare'           
           print 'gen'
           print 'compute'
           #print outball                          
           outmap.append(results)           
        output=T.reshape(outmap,(self.batchsize,1,self.height,self.width))
        return output

    def _loopoverallball(self, ballid,batchid):        
        ox=self.middle[batchid][ballid*2].reshape((1,1))
        print "ox:",ox.ndim
        x=T.tile(ox,(self.height,self.width))
        oy=self.middle[batchid][ballid*2+1].reshape((1,1))
        y=T.tile(oy,(self.height,self.width))
        w=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
        h=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
        cof=(T.pow(x-w,2)+T.pow(y-h,2))*(-1.0/self.sigma)        
        print T.exp(cof).ndim
        return T.exp(cof)
        #return 1.0/(1+T.exp(-((T.exp(cof))-10.0)))
 
#####Ball Model

class BallModel(Layer):
    def __init__(self,height,width,batchsize):
        super(BallModel,self).__init__()
        self.input=T.matrix()
        self.height=height
        self.width=width
        #self.prenet=prenet
        #self.input=prenet.input
        self.batchsize=batchsize
    
    def _getrx(self,d):
        r=theano.shared(np.zeros((3,3),dtype='float32'))
        r=T.set_subtensor(r[0,0],1.0)
        r=T.set_subtensor(r[1,1],T.cos(d))
        r=T.set_subtensor(r[1,2],-T.sin(d))
        r=T.set_subtensor(r[2,1],T.sin(d))
        r=T.set_subtensor(r[2,2],T.cos(d))        
        return r

    def _getry(self,d):
        r=theano.shared(np.zeros((3,3),dtype='float32'))
        r=T.set_subtensor(r[1,1],1.0)
        r=T.set_subtensor(r[0,0],T.cos(d))
        r=T.set_subtensor(r[0,2],-T.sin(d))
        r=T.set_subtensor(r[2,0],T.sin(d))
        r=T.set_subtensor(r[2,2],T.cos(d))        
        return r

    def _getrz(self,d):
        r=theano.shared(np.zeros((3,3),dtype='float32'))
        r=T.set_subtensor(r[2,2],1.0)
        r=T.set_subtensor(r[0,0],T.cos(d))
        r=T.set_subtensor(r[0,1],-T.sin(d))
        r=T.set_subtensor(r[1,0],T.sin(d))
        r=T.set_subtensor(r[1,1],T.cos(d))        
        return r

    def get_output(self, train=False):        
        initial=[[27,0,15],
                [9,0,10],
                [-9,0,10],
                [-27,0,10],
                [27,0,30],
                [9,0,30],
                [-9,0,30],
                [-27,0,30],
                [27,0,50],
                [9,0,50],
                [-9,0,50],
                [-27,0,50],
                [28,0,70],
                [8,0,70],
                [-11,0,70],
                [-28,0,70],
                [41,0,15],
                [45,0,15],
                [49,0,15],
                [53,0,15],
                [67,0,15],
                [81,0,15],
                [94,0,15],
                [105,0,15],
                [28,0,88],
                [28,0,104],
                [28,0,119],
                [28,0,133],
                [28,0,145],
                [28,0,156],
                [8,0,90],
                [8,0,110],
                [8,0,127],
                [8,0,141],
                [8,0,155],
                [8,0,169],
                [-11,0,88],
                [-11,0,104],
                [-11,0,119],
                [-11,0,133],
                [-11,0,145],
                [-11,0,156],
                [-28,0,85],
                [-28,0,95],
                [-28,0,105],
                [-28,0,116],
                [-28,0,127],
                [-28,0,135]]
        outputans=[]
        X = self.get_input(train)
        
        for batch in range(0,self.batchsize):
            d=X[batch].reshape((26,)) ##ndim=0

            ro=T.dot(T.dot(self._getrx(d[3]),self._getry(d[4])),self._getrz(d[5]))
            to1to16=theano.shared(np.zeros((3,1),dtype='float32'))
            ###o'=(d1,d2,d3)
            to1to16=T.set_subtensor(to1to16[0],d[0])
            to1to16=T.set_subtensor(to1to16[1],d[1])
            to1ti16=T.set_subtensor(to1to16[2],d[2])
            o1to16=T.tile(to1ti16,(1,16))
            #
            xi1to16=theano.shared(np.zeros((16,3),dtype='float32'))
            xi1to16=T.dot(ro,np.array(initial[0:16]).T)+o1to16

            ####################################Thumb First
            ###17-20
            r16=T.dot(self._getry(d[6]),self._getrz(d[7]))
            ###x16origin rotation joint is sphere 1
            txori16=theano.shared(np.zeros((3,1),dtype='float32'))
            txori16=T.set_subtensor(txori16[0],initial[0][0])
            txori16=T.set_subtensor(txori16[1],initial[0][1])
            txori16=T.set_subtensor(txori16[2],initial[0][2])
            xori16=T.tile(txori16,(1,4))

            to17to20=theano.shared(np.zeros((3,1),dtype='float32'))
            to17to20=T.set_subtensor(to17to20[0],xi1to16[0][0])
            to17to20=T.set_subtensor(to17to20[1],xi1to16[1][0])
            to17to20=T.set_subtensor(to17to20[2],xi1to16[2][0])
            o17to20=T.tile(to17to20,(1,4))
            xi17to20=T.dot(T.dot(ro,r16),np.array(initial[16:20]).T -xori16)+o17to20

            ######################################Thumb Second
            ###21-22
            r20=self._getrz(d[8])
            txori20=theano.shared(np.zeros((3,1),dtype='float32'))
            txori20=T.set_subtensor(txori20[0],initial[19][0])
            txori20=T.set_subtensor(txori20[1],initial[19][1])
            txori20=T.set_subtensor(txori20[2],initial[19][2])
            xori20=T.tile(txori20,(1,2))

            to21to22=theano.shared(np.zeros((3,1),dtype='float32'))
            to21to22=T.set_subtensor(to21to22[0],xi17to20[0][3])
            to21to22=T.set_subtensor(to21to22[1],xi17to20[1][3])
            to21to22=T.set_subtensor(to21to22[2],xi17to20[2][3])
            o21to22=T.tile(to21to22,(1,2))
            xi21to22=T.dot(T.dot(T.dot(ro,r16),r20) ,np.array(initial[20:22]).T -xori20)+o21to22

            ######################################Thumb Third
            ###23-24
            r22=self._getrz(d[9])
            txori22=theano.shared(np.zeros((3,1),dtype='float32'))
            txori22=T.set_subtensor(txori22[0],initial[21][0])
            txori22=T.set_subtensor(txori22[1],initial[21][1])
            txori22=T.set_subtensor(txori22[2],initial[21][2])
            xori22=T.tile(txori22,(1,2))

            to23to24=theano.shared(np.zeros((3,1),dtype='float32'))
            to23to24=T.set_subtensor(to23to24[0],xi21to22[0][1])
            to23to24=T.set_subtensor(to23to24[1],xi21to22[1][1])
            to23to24=T.set_subtensor(to23to24[2],xi21to22[2][1])
            o23to24=T.tile(to23to24,(1,2))
            xi23to24=T.dot(T.dot(T.dot(T.dot(ro,r16),r20),r22) ,np.array(initial[22:24]).T -xori22)+o23to24

            ######################################First Joint Index
            ###25-26
            r13=T.dot(self._getrx(d[10]),self._getry(d[11]))
            txori13=theano.shared(np.zeros((3,1),dtype='float32'))
            txori13=T.set_subtensor(txori13[0],initial[12][0])
            txori13=T.set_subtensor(txori13[1],initial[12][1])
            txori13=T.set_subtensor(txori13[2],initial[12][2])
            xori13=T.tile(txori13,(1,2))

            to25to26=theano.shared(np.zeros((3,1),dtype='float32'))
            to25to26=T.set_subtensor(to25to26[0],xi1to16[0][12])
            to25to26=T.set_subtensor(to25to26[1],xi1to16[1][12])
            to25to26=T.set_subtensor(to25to26[2],xi1to16[2][12])
            o25to26=T.tile(to25to26,(1,2))
            xi25to26=T.dot(T.dot(ro,r13),np.array(initial[24:26]).T -xori13)+o25to26

            ######################################First Joint Middle
            ###31-32
            r14=T.dot(self._getrx(d[14]),self._getry(d[15]))
            txori14=theano.shared(np.zeros((3,1),dtype='float32'))
            txori14=T.set_subtensor(txori14[0],initial[13][0])
            txori14=T.set_subtensor(txori14[1],initial[13][1])
            txori14=T.set_subtensor(txori14[2],initial[13][2])
            xori14=T.tile(txori14,(1,2))

            to31to32=theano.shared(np.zeros((3,1),dtype='float32'))
            to31to32=T.set_subtensor(to31to32[0],xi1to16[0][13])
            to31to32=T.set_subtensor(to31to32[1],xi1to16[1][13])
            to31to32=T.set_subtensor(to31to32[2],xi1to16[2][13])
            o31to32=T.tile(to31to32,(1,2))
            xi31to32=T.dot(T.dot(ro,r14),np.array(initial[30:32]).T -xori14)+o31to32

            ######################################First Joint Ring
            ###37-38
            r15=T.dot(self._getrx(d[18]),self._getry(d[19]))
            txori15=theano.shared(np.zeros((3,1),dtype='float32'))
            txori15=T.set_subtensor(txori15[0],initial[14][0])
            txori15=T.set_subtensor(txori15[1],initial[14][1])
            txori15=T.set_subtensor(txori15[2],initial[14][2])
            xori15=T.tile(txori15,(1,2))

            to37to38=theano.shared(np.zeros((3,1),dtype='float32'))
            to37to38=T.set_subtensor(to37to38[0],xi1to16[0][14])
            to37to38=T.set_subtensor(to37to38[1],xi1to16[1][14])
            to37to38=T.set_subtensor(to37to38[2],xi1to16[2][14])
            o37to38=T.tile(to37to38,(1,2))
            xi37to38=T.dot(T.dot(ro,r15),np.array(initial[36:38]).T -xori15)+o37to38

            ######################################First Joint Little
            ###43-44
            r16=T.dot(self._getrx(d[22]),self._getry(d[23]))
            txori16=theano.shared(np.zeros((3,1),dtype='float32'))
            txori16=T.set_subtensor(txori16[0],initial[15][0])
            txori16=T.set_subtensor(txori16[1],initial[15][1])
            txori16=T.set_subtensor(txori16[2],initial[15][2])
            xori16=T.tile(txori16,(1,2))

            to43to44=theano.shared(np.zeros((3,1),dtype='float32'))
            to43to44=T.set_subtensor(to43to44[0],xi1to16[0][15])
            to43to44=T.set_subtensor(to43to44[1],xi1to16[1][15])
            to43to44=T.set_subtensor(to43to44[2],xi1to16[2][15])
            o43to44=T.tile(to43to44,(1,2))
            xi43to44=T.dot(T.dot(ro,r16),np.array(initial[42:44]).T -xori16)+o43to44

            ######################################Second Joint Index
            ###27-28
            r26=self._getrx(d[12])
            txori26=theano.shared(np.zeros((3,1),dtype='float32'))
            txori26=T.set_subtensor(txori26[0],initial[25][0])
            txori26=T.set_subtensor(txori26[1],initial[25][1])
            txori26=T.set_subtensor(txori26[2],initial[25][2])
            xori26=T.tile(txori26,(1,2))

            to27to28=theano.shared(np.zeros((3,1),dtype='float32'))
            to27to28=T.set_subtensor(to27to28[0],xi25to26[0][1])
            to27to28=T.set_subtensor(to27to28[1],xi25to26[1][1])
            to27to28=T.set_subtensor(to27to28[2],xi25to26[2][1])
            o27to28=T.tile(to27to28,(1,2))
            xi27to28=T.dot(T.dot(T.dot(ro,r13),r26),np.array(initial[26:28]).T -xori26)+o27to28

            ######################################Second Joint Middle
            ###33-34
            r32=self._getrx(d[16])
            txori32=theano.shared(np.zeros((3,1),dtype='float32'))
            txori32=T.set_subtensor(txori32[0],initial[31][0])
            txori32=T.set_subtensor(txori32[1],initial[31][1])
            txori32=T.set_subtensor(txori32[2],initial[31][2])
            xori32=T.tile(txori32,(1,2))

            to33to34=theano.shared(np.zeros((3,1),dtype='float32'))
            to33to34=T.set_subtensor(to33to34[0],xi31to32[0][1])
            to33to34=T.set_subtensor(to33to34[1],xi31to32[1][1])
            to33to34=T.set_subtensor(to33to34[2],xi31to32[2][1])
            o33to34=T.tile(to33to34,(1,2))
            xi33to34=T.dot(T.dot(T.dot(ro,r14),r32),np.array(initial[32:34]).T -xori32)+o33to34

            ######################################Second Joint Ring
            ###39-40
            r38=self._getrx(d[20])
            txori38=theano.shared(np.zeros((3,1),dtype='float32'))
            txori38=T.set_subtensor(txori38[0],initial[37][0])
            txori38=T.set_subtensor(txori38[1],initial[37][1])
            txori38=T.set_subtensor(txori38[2],initial[37][2])
            xori38=T.tile(txori38,(1,2))

            to39to40=theano.shared(np.zeros((3,1),dtype='float32'))
            to39to40=T.set_subtensor(to39to40[0],xi37to38[0][1])
            to39to40=T.set_subtensor(to39to40[1],xi37to38[1][1])
            to39to40=T.set_subtensor(to39to40[2],xi37to38[2][1])
            o39to40=T.tile(to39to40,(1,2))
            xi39to40=T.dot(T.dot(T.dot(ro,r15),r38),np.array(initial[38:40]).T -xori38)+o39to40

            ######################################Second Joint Little
            ###45-46
            r44=self._getrx(d[24])
            txori44=theano.shared(np.zeros((3,1),dtype='float32'))
            txori44=T.set_subtensor(txori44[0],initial[43][0])
            txori44=T.set_subtensor(txori44[1],initial[43][1])
            txori44=T.set_subtensor(txori44[2],initial[43][2])
            xori44=T.tile(txori44,(1,2))

            to45to46=theano.shared(np.zeros((3,1),dtype='float32'))
            to45to46=T.set_subtensor(to45to46[0],xi43to44[0][1])
            to45to46=T.set_subtensor(to45to46[1],xi43to44[1][1])
            to45to46=T.set_subtensor(to45to46[2],xi43to44[2][1])
            o45to46=T.tile(to45to46,(1,2))
            xi45to46=T.dot(T.dot(T.dot(ro,r16),r44),np.array(initial[44:46]).T -xori44)+o45to46

            #####################################Third Joint Index
            ###29-30
            r28=self._getrx(d[13])
            txori28=theano.shared(np.zeros((3,1),dtype='float32'))
            txori28=T.set_subtensor(txori28[0],initial[27][0])
            txori28=T.set_subtensor(txori28[1],initial[27][1])
            txori28=T.set_subtensor(txori28[2],initial[27][2])
            xori28=T.tile(txori28,(1,2))

            to29to30=theano.shared(np.zeros((3,1),dtype='float32'))
            to29to30=T.set_subtensor(to29to30[0],xi27to28[0][1])
            to29to30=T.set_subtensor(to29to30[1],xi27to28[1][1])
            to29to30=T.set_subtensor(to29to30[2],xi27to28[2][1])
            o29to30=T.tile(to29to30,(1,2))
            xi29to30=T.dot(T.dot(T.dot(T.dot(ro,r13),r26),r28),np.array(initial[28:30]).T -xori28)+o29to30

            #####################################Third Joint Middle
            ###35-36
            r34=self._getrx(d[17])
            txori34=theano.shared(np.zeros((3,1),dtype='float32'))
            txori34=T.set_subtensor(txori34[0],initial[33][0])
            txori34=T.set_subtensor(txori34[1],initial[33][1])
            txori34=T.set_subtensor(txori34[2],initial[33][2])
            xori34=T.tile(txori34,(1,2))

            to35to36=theano.shared(np.zeros((3,1),dtype='float32'))
            to35to36=T.set_subtensor(to35to36[0],xi33to34[0][1])
            to35to36=T.set_subtensor(to35to36[1],xi33to34[1][1])
            to35to36=T.set_subtensor(to35to36[2],xi33to34[2][1])
            o35to36=T.tile(to35to36,(1,2))
            xi35to36=T.dot(T.dot(T.dot(T.dot(ro,r14),r32),r34),np.array(initial[34:36]).T -xori34)+o35to36

            #####################################Third Joint Ring
            ###41-42
            r40=self._getrx(d[21])
            txori40=theano.shared(np.zeros((3,1),dtype='float32'))
            txori40=T.set_subtensor(txori40[0],initial[39][0])
            txori40=T.set_subtensor(txori40[1],initial[39][1])
            txori40=T.set_subtensor(txori40[2],initial[39][2])
            xori40=T.tile(txori40,(1,2))

            to41to42=theano.shared(np.zeros((3,1),dtype='float32'))
            to41to42=T.set_subtensor(to41to42[0],xi39to40[0][1])
            to41to42=T.set_subtensor(to41to42[1],xi39to40[1][1])
            to41to42=T.set_subtensor(to41to42[2],xi39to40[2][1])
            o41to42=T.tile(to41to42,(1,2))
            xi41to42=T.dot(T.dot(T.dot(T.dot(ro,r15),r38),r40),np.array(initial[40:42]).T -xori40)+o41to42

            #####################################Third Joint Little
            ###47-48
            r46=self._getrx(d[25])
            txori46=theano.shared(np.zeros((3,1),dtype='float32'))
            txori46=T.set_subtensor(txori46[0],initial[45][0])
            txori46=T.set_subtensor(txori46[1],initial[45][1])
            txori46=T.set_subtensor(txori46[2],initial[45][2])
            xori46=T.tile(txori46,(1,2))

            to47to48=theano.shared(np.zeros((3,1),dtype='float32'))
            to47to48=T.set_subtensor(to47to48[0],xi45to46[0][1])
            to47to48=T.set_subtensor(to47to48[1],xi45to46[1][1])
            to47to48=T.set_subtensor(to47to48[2],xi45to46[2][1])
            o47to48=T.tile(to47to48,(1,2))
            xi47to48=T.dot(T.dot(T.dot(T.dot(ro,r16),r44),r46),np.array(initial[46:48]).T -xori46)+o47to48

            ret=T.concatenate([xi1to16 ,xi17to20,xi21to22,xi23to24,xi25to26,xi27to28,xi29to30,xi31to32,xi33to34,xi35to36,xi37to38,xi39to40,
                    xi41to42,xi43to44,xi45to46,xi47to48],axis=1)
            ret=ret.reshape((144,))
            ans=[]
            #ans=theano.shared(np.zeros((96,),dtype='float32'))
            for k in range(0,48):
                ans.append(( ret[3*k]   /-ret[3*k+2]+0.5)*self.width)
                ans.append((-ret[3*k+1] /-ret[3*k+2]+0.5)*self.height)
                ans.append(-ret[3*k+2])
                #ans=T.set_subtensor(ans[2*k],  ( ret[3*k]   /-ret[3*k+2]+0.5)*self.width)
                #ans=T.set_subtensor(ans[2*k+1],(-ret[3*k+1] /-ret[3*k+2]+0.5)*self.height)
            #print ans,ans.ndim
            ans=T.reshape(ans,(144,))
            print ans,ans.ndim
            outputans.append(ans)
        outputans=T.reshape(outputans,(self.batchsize,144))
        return outputans

###XYZ - XY
class ConvertToXY(Layer):
    def __init__(self,batchsize,height,width,ballnum):
        super(ConvertToXY,self).__init__()
        self.input=T.matrix
        self.batchsize=batchsize
        self.height=height
        self.width=width
        self.ballnum=ballnum

    def get_output(self,train=False):
        outputans=[]
        XX = self.get_input(train)
        for batch in range(0,self.batchsize):
            X=XX[batch]
            for k in range(0,self.ballnum):
                outputans.append(( X[3*k]  /-X[3*k+2]+0.5)*self.width)
                outputans.append((-X[3*k+1]/-X[3*k+2]+0.5)*self.height)
        outputans=T.reshape(outputans,(self.batchsize,96))
        return outputans
