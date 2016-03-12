import numpy as np
import theano
import theano.tensor as T
import math
from keras.layers.core import Layer

floatX = theano.config.floatX

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
        self.middle = self.get_input(train)
        ##########Main Function        
        resultskin=theano.shared(np.zeros((self.batchsize,60,self.height,self.width),dtype='float32'))
        resultdepth=theano.shared(np.zeros((self.batchsize,60,self.height,self.width),dtype='float32'))
        minz=self.middle[:,120:180].min(axis=1)
        for i in range(0,60):
            ox=100.0*self.middle[:,(i)].reshape((self.batchsize,1))            
            x=T.tile(ox,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
            oy=100.0*self.middle[:,(i)+60].reshape((self.batchsize,1))
            y=T.tile(oy,(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
            w=T.tile(T.tile(T.arange(0,self.width),(self.height,)),(self.batchsize,)).reshape((self.batchsize,self.height,self.width))
            h=T.tile(T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width)),(self.batchsize,1)).reshape((self.batchsize,self.height,self.width))
            resultskin=T.set_subtensor(resultskin[:,i,:,:],T.exp((T.minimum(T.pow(x-w,2)+T.pow(y-h,2),140))*(-1.0/self.sigma)))
            multiply=T.tile((self.middle[:,(i)+120]-minz+0.2).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,self.height,self.width))
            resultdepth=T.set_subtensor(resultdepth[:,i,:,:],multiply*T.exp((T.minimum(T.pow(x-w,2)+T.pow(y-h,2),140))*(-1.0/self.sigma)))
        
        resultskin=resultskin.max(axis=[1])
        resultdepth=resultdepth.max(axis=[1])

        minx=T.tile(resultdepth.min(axis=[1,2]).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,(self.height*self.width))).reshape((self.batchsize,self.height,self.width))
        maxx=T.tile(resultdepth.max(axis=[1,2]).reshape((self.batchsize,1)),(1,self.height*self.width)).reshape((self.batchsize,(self.height*self.width))).reshape((self.batchsize,self.height,self.width))

        resultdepth=(resultdepth-minx)/(maxx-minx)
           
        print 'prepare'           
        print 'gen'
        print 'compute'
        
        output=theano.shared(np.zeros((self.batchsize,2,self.height,self.width)))
        output=T.set_subtensor(output[:,0,:,:],resultskin)
        output=T.set_subtensor(output[:,1,:,:],resultdepth)
        output=T.reshape(output,(self.batchsize,2*self.height*self.width))
        #output=T.reshape(results,(self.batchsize,1,self.height,self.width))
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
    def _loopoverallbatch(self,batch):
        initial=np.array([[  0.10,0.18,0.0  ],
                          [  0.05,0.18,0.034],
                          [  0.00,0.18,0.030],
                          [- 0.05,0.18,0.035],
                          [- 0.11,0.18,0.040],

                          [ 0.118,0.25,0.030],
                          [  0.06,0.25,0.036],
                          [  0.00,0.25,0.029],
                          [ -0.06,0.25,0.043],
                          [-0.118,0.25,0.026],

                          [ 0.126,0.32,0.0295],
                          [ 0.065,0.32,0.024677],
                          [  0.00,0.32,0.0204134],
                          [-0.065,0.32,0.0453488],
                          [-0.126,0.32,0.0412145],

                          [ 0.130,0.39,0.0122739],
                          [  0.07,0.39,0.0121477],
                          [  0.00,0.39,0.0116279],
                          [ -0.07,0.39,0.0390181],
                          [-0.130,0.39,0.0483204],

                          [ 0.132,0.46,0.00530],
                          [ 0.075,0.46,0.00723],
                          [  0.00,0.46,0.0129199],
                          [-0.075,0.46,0.0378553],
                          [-0.132,0.46,0.05],

                          [ 0.134,0.53,0.00387],
                          [ 0.065,0.53,0.0118863],
                          [  0.00,0.53,0.0197674],
                          [-0.065,0.53,0.0387597],

                          [  0.00,0.12,0.03],
                          [ -0.06,0.12,0.033],
                          [ -0.10,0.12,0.037],
                          
                          [ -0.03,0.07,0.032],
                          [ -0.08,0.07,0.036],
                          [ -0.13,0.07,0.041],

                          [  0.12,0.25,0.0],
                          [  0.16,0.25,0.0],
                          [  0.21,0.25,0.0],
                          [  0.26,0.25,0.0],
                          [  0.32,0.25,0.0],
                          [  0.38,0.25,0.0],

                          [  0.14,0.58,0.00387],
                          [  0.15,0.64,0.0040],
                          [ 0.155,0.70,0.00428],
                          [ 0.165,0.75,0.004789],
                          [  0.17,0.80,0.00493425],

                          [  0.03,0.58,0.0118863],
                          [ 0.036,0.65,0.0106],
                          [ 0.042,0.72,0.0121],
                          [  0.05,0.79,0.0133],
                          [ 0.055,0.84,0.0149],

                          [ -0.05,0.58,0.0197],
                          [-0.048,0.63,0.0203],
                          [-0.045,0.69,0.0233],
                          [ -0.04,0.75,0.0253 ],
                          [-0.035,0.80,0.0270],

                          [ -0.14,0.54,0.0419897],
                          [-0.153,0.58,0.0432],
                          [ -0.17,0.63,0.0419],
                          [-0.175,0.68,0.0410]])

        d=self.middle[batch].reshape((26,)) ##ndim=0        

        rpalm=T.dot(T.dot(self._getrx(d[3]),self._getry(d[4])),self._getrz(d[5]))

        x=theano.shared(np.zeros((3,60),dtype='float32'))
        
        for i in range(0,35):
            
            x=T.set_subtensor(x[:,i],T.dot(rpalm,initial[i][:].T)+d[0:3])

        othumb=x[:,5]
        rthumb=T.dot(self._getry(d[6]),self._getrz(d[7]))
        for i in range(35,37):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(rpalm,rthumb),(initial[i][:]-initial[5][:]).T)+othumb)

        othumbsecond=x[:,36]        
        rthumbsecond=self._getrz(d[8])
        for i in range(37,39):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(rpalm,rthumb),rthumbsecond),(initial[i][:]-initial[36][:]).T)+othumbsecond)

        othumbthird=x[:,38]
        rthumbthird=self._getrz(d[9])
        for i in range(39,41):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(T.dot(rpalm,rthumb),rthumbsecond),rthumbthird),(initial[i][:]-initial[38][:]).T)+othumbthird)

        for k in range(0,3):
            pfirst=25+k
            ofirst=x[:,pfirst]
            rfirst=T.dot(self._getrx(d[11+4*k-1]),self._getrz(d[12+4*k-1]))
            for i in range(42+5*k-1,42+5*k+1):
                x=T.set_subtensor(x[:,i],T.dot(T.dot(rpalm,rfirst),(initial[i][:]-initial[pfirst][:]).T)+ofirst)    
        
            psecond=42+5*k
            osecond=x[:,psecond]
            rsecond=self._getrx(d[13+4*k-1])
            for i in range(44+5*k-1,44+5*k+1):
                x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(rpalm,rfirst),rsecond),(initial[i][:]-initial[psecond][:]).T)+osecond)

            pthird=44+5*k
            othird=x[:,pthird]
            rthird=self._getrx(d[13+4*k])
            i=45+5*k
            x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(T.dot(rpalm,rfirst),rsecond),rthird),(initial[i][:]-initial[pthird][:]).T)+othird)            

        p=24
        o=x[:,p]
        rlittle=T.dot(self._getrx(d[22]),self._getrz(d[23]))
        i=56
        x=T.set_subtensor(x[:,i],T.dot(T.dot(rpalm,rlittle),(initial[i][:]-initial[p][:]).T)+o)

        p=56
        o=x[:,p]
        rlittlesecond=self._getrx(d[24])
        i=57
        x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(rpalm,rlittle),rlittlesecond),(initial[i][:]-initial[p][:]).T)+o)
        
        p=57
        o=x[:,p]
        rlittlethird=self._getrx(d[25])
        for i in range(58,60):
            x=T.set_subtensor(x[:,i],T.dot(T.dot(T.dot(T.dot(rpalm,rlittle),rlittlesecond),rlittlethird),(initial[i][:]-initial[p][:]).T)+o)
        
        toadd=np.array([ [-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085,-0.085],
                         [-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43,-0.43],
                         [ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])


        x=x+toadd
        

        ans=theano.shared(np.zeros((3,60),dtype='float32'))
        ans=T.set_subtensor(ans[0,:],T.maximum(-10,T.minimum((x[0,:]/T.maximum(x[2,:],0.1) +0.5)*self.width,105))/100.0)
        ans=T.set_subtensor(ans[1,:],T.maximum(-10,T.minimum((-x[1,:]/T.maximum(x[2,:],0.1)+0.5)*self.height,105))/100.0)
        ans=T.set_subtensor(ans[2,:],x[2,:])
        #ans=ans.T
        ans=T.reshape(ans,(180,))
        return ans

    def get_output(self, train=False):        
        
        outputans=[]
        X = self.get_input(train)
        self.middle=X

        results, updates = theano.scan(fn=self._loopoverallbatch ,
                                                           sequences=[T.arange(0,self.batchsize)])           

        print results,results.ndim
        return results
      