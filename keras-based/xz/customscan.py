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
    def _loopoverallballallbatch(self, ballid): 
        ox=self.middle[:,(ballid)*3].reshape((self.batchsize,1))
        x=T.tile(ox,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        oy=self.middle[:,(ballid)*3+1].reshape((self.batchsize,1))
        y=T.tile(oy,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        w=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
        h=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
        multiply=T.tile(self.middle[:,(ballid)*3+2].reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
        results=multiply*T.exp((T.pow(x-w,2)+T.pow(y-h,2))*(-1.0/self.sigma))
        return results
    def get_output(self, train=False):
        #X=self.get_input(train)        
        #outball=apply_model(self.nownet,X)    
        outball=self.get_input(train)
        self.middle = outball
        ##########Main Function
        #Read Data        
        outmap=[]        
        results, updates = theano.scan(fn=self._loopoverallballallbatch ,
                                                           sequences=[T.arange(0,self.ballnum)])           
        results=results.sum(axis=[0])        

        minx=T.tile(results.min(axis=[1,2]).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,(self.height*self.width))).reshape((self.batchsize,self.height,self.width))
        maxx=T.tile(results.max(axis=[1,2]).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,(self.height*self.width))).reshape((self.batchsize,self.height,self.width))

        results=(results-minx)/(maxx-minx)
           
        print 'prepare'           
        print 'gen'
        print 'compute'
        
        output=T.reshape(results,(self.batchsize,1,self.height,self.width))
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
