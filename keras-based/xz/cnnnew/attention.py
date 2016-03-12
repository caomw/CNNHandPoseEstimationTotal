import numpy as np
import theano
import theano.tensor as T
import math
from keras.layers.core import Layer

from ..utils import apply_model
floatX = theano.config.floatX


class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
hhhhhh    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    """
    def __init__(self,
                 localization_net,
                 downsample_factor=1,weights=None,
                 return_theta=False,
                 **kwargs):
        super(SpatialTransformer, self).__init__()
        self.downsample_factor = downsample_factor
        self.locnet = localization_net
        self.params=[]
#        self.params = localization_net.params
        self.regularizers = localization_net.regularizers
        self.constraints = localization_net.constraints
        self.input = self.locnet.input  # This must be T.tensor4()
        print self.input
        self.return_theta = return_theta
        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train=False):
        X = self.get_input(train)
        print X
        #theta = apply_model(self.locnet, X)
        print 'now get output'
        theta=self.locnet.get_output(X)
        print '3',theta[0],theta
        theta = theta.reshape((X.shape[0], 2, 3))
        output = self._transform(theta, X, self.downsample_factor)
        print output
        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output

    def _repeat(self, x, n_repeats):
        rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
        x = T.dot(x.reshape((-1, 1)), rep)        
        return x.flatten()

    def _interpolate(self, im, x, y, downsample_factor):
        # constants
        num_batch, height, width, channels = im.shape
        height_f = T.cast(height, floatX)
        width_f = T.cast(width, floatX)
        out_height = T.cast(height_f // downsample_factor, 'int64')
        out_width = T.cast(width_f // downsample_factor, 'int64')
        zero = T.zeros([], dtype='int64')
        max_y = T.cast(im.shape[1] - 1, 'int64')
        max_x = T.cast(im.shape[2] - 1, 'int64')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = T.cast(T.floor(x), 'int64')
        x1 = x0 + 1
        y0 = T.cast(T.floor(y), 'int64')
        y1 = y0 + 1

        x0 = T.clip(x0, zero, max_x)
        x1 = T.clip(x1, zero, max_x)
        y0 = T.clip(y0, zero, max_y)
        y1 = T.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = self._repeat(
            T.arange(num_batch, dtype='int32')*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat
        #  image and restore channels dim
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finanly calculate interpolated values
        x0_f = T.cast(x0, floatX)
        x1_f = T.cast(x1, floatX)
        y0_f = T.cast(y0, floatX)
        y1_f = T.cast(y1, floatX)
        wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
        wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
        wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
        wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
        output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
        return output

    def _linspace(self, start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        start = T.cast(start, floatX)
        stop = T.cast(stop, floatX)
        num = T.cast(num, floatX)
        step = (stop-start)/(num-1)
        return T.arange(num, dtype=floatX)*step+start

    def _meshgrid(self, height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = T.dot(T.ones((height, 1)),
                    self._linspace(-1.0, 1.0, width).dimshuffle('x', 0))
        y_t = T.dot(self._linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                    T.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = T.ones_like(x_t_flat)
        grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _transform(self, theta, input, downsample_factor):
        num_batch, num_channels, height, width = input.shape
        theta = T.reshape(theta, (-1, 2, 3))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = T.cast(height, floatX)
        width_f = T.cast(width, floatX)
        out_height = T.cast(height_f // downsample_factor, 'int64')
        out_width = T.cast(width_f // downsample_factor, 'int64')
        grid = self._meshgrid(out_height, out_width)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = T.dot(theta, grid)
        x_s, y_s = T_g[:, 0], T_g[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()

        # dimshuffle input to  (bs, height, width, channels)
        input_dim = input.dimshuffle(0, 2, 3, 1)
        input_transformed = self._interpolate(
            input_dim, x_s_flat, y_s_flat,
            downsample_factor)

        output = T.reshape(input_transformed,
                           (num_batch, out_height, out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)
        return output


class AttentionST(SpatialTransformer):
    '''
    A Spatial Transformer limitted to scaling,
    cropping and translation.
    '''
    def __init__(self, *args, **kwargs):
        super(AttentionST, self).__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input()
        # locnet.get_output(X) should be shape (batchsize, 6)
        mask = np.ones((2, 3))
        mask[1, 0] = 0
        mask[0, 1] = 0
        mask = theano.shared(mask.astype(floatX))
        theta = self.locnet.get_output(X).reshape((X.shape[0], 2, 3))
        theta = theta * mask[None, :, :]

        output = self._transform(theta, X, self.downsample_factor)
        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output


class ST2(Layer):
    '''This implementation is similar to the equations in the paper
    but uses a lot more memory
    '''
    def __init__(self,
                 localization_net,
                 img_shape,
                 downsample_factor=(1, 1),
                 return_theta=False,
                 **kwargs):
        super(ST2, self).__init__()
        self.ds = downsample_factor
        self.locnet = localization_net
        self.img_shape = img_shape
        self.params = localization_net.params
        self.regularizers = localization_net.regularizers
        self.constraints = localization_net.constraints
        self.input = localization_net.input  # this should be T.tensor4()
        self.return_theta = return_theta

    def get_output(self, train=False):
        X = self.get_input()
        # locnet.get_output(X) should be shape (batchsize, 6)
        theta = self.locnet.get_output(train)  # .reshape((X.shape[0], 2, 3))
        thetas = T.nnet.sigmoid(theta[:, :4])
        thetat = T.nnet.sigmoid(theta[:, 4:]) * X.shape[2]
        theta = T.concatenate([thetas.reshape((X.shape[0], 2, 2)),
                               thetat.reshape((X.shape[0], 2, 1))],
                              axis=2)

        output = self._transform(X, theta, self.ds)
        if self.return_theta:
            return theta.reshape((X.shape[0], 6))
        else:
            return output

    def _meshgrid(self, row, col):
        x, y = np.meshgrid(np.linspace(0, row-1, row),
                           np.linspace(0, col-1, col))
        # x, y = np.meshgrid(np.linspace(-1, 1, row),
        #                   np.linspace(-1, 1, col))
        X = theano.shared(x.astype(floatX))
        Y = theano.shared(y.astype(floatX))
        ones = T.ones_like(X)
        grid = T.concatenate([X[None, :, :], Y[None, :, :], ones[None, :, :]],
                             axis=0)
        return grid

    def _transform(self, X, theta, ds):
        b = X.shape[0]
        chan, row, col = self.img_shape
        new_row = row / ds[0]
        new_col = col / ds[1]
        grid = self._meshgrid(new_row, new_col)
        new_grid = T.tensordot(theta, grid.reshape((3, new_row*new_col)),
                               axes=(2, 0))
        output = []
        for i in range(chan):
            out = X[:, i, :, :, None] * T.maximum(
                0, 1 - abs(new_grid[:, None, None, 0, :] -
                           grid[None, 0, :, :, None])) * T.maximum(
                               0, 1 - abs(new_grid[:, None, None, 1, :]
                                          - grid[None, 1, :, :, None]))
            out = out.sum(axis=(1, 2)).reshape((b, row, col))
            output.append(out.reshape((b, new_row,
                                       new_col)).dimshuffle(0, 'x', 1, 2))
        output = T.concatenate(output, axis=1)
        return output


####Gaussian Hand Model
class GaussianModelori(Layer):
    def __init__(self,prenet,height,width,ballnum,sigma,batchsize):
       super(GaussianModelori,self).__init__()
       self.input=prenet.input
       self.nownet=prenet
       self.middle=[prenet.get_output(prenet.input)    ]
       self.params=prenet.params
       self.height=height
       self.width=width
       self.ballnum=ballnum
       self.sigma=sigma
       self.batchsize=batchsize

    def get_output(self, train=False):
        X=self.get_input(train)        
        #outball=apply_model(self.nownet,X)    
        outball=self.middle
        ##########Main Function
        #Read Data        
        outmap=[]
        a=np.zeros((self.height*self.width),dtype="int32")
        b=np.zeros((self.height*self.width),dtype="int32")                
        c=np.zeros((self.height*self.width),dtype="int32")                
        for batch in range(0,self.batchsize):
           for i in range(0,self.height):
              for j in range(0,self.width):
                 a[i*self.width+j]=j  #col     axis x
                 b[i*self.width+j]=i  #row   axis y                            
                 c[i*self.width+j]=batch
           colshare=theano.shared(np.array(a).T)
           rowshare=theano.shared(np.array(b).T)
           batchshare=theano.shared(np.array(c).T)
           results, updates = theano.scan(fn=self._loopoverallball ,
                                                           sequences=[colshare,rowshare,batchshare])
           print results
           print 'prepare'           
           print 'gen'
           print 'compute'
           print outball                          
           outmap.append(results)           
        output=T.reshape(outmap,(self.batchsize,1,self.height,self.width))
        return output

    def _loopoverallball(self, h,w,bid):
        ball=self.middle[bid]
        [results,updates]=theano.scan(lambda id,ball: T.exp(-1.0/self.sigma* (  (( ball[id*2]   +0.5)  * self.width -w)  * ((ball[id*2+1] +0.5)  * self.width-w) 
                                                                                                                                        + ((ball[id*2]  +  0.5)  * self.height-h)  * ((ball[id*2+1] +0.5)  * self.height-h) ) ),
                                                                    sequences=[np.arange(0,self.ballnum)],
                                                                    non_sequences=[ball])   
        return results.sum()


#####XYZ->XY

class GaussianModel(Layer):
    def __init__(self,prenet,height,width,ballnum,sigma,batchsize):
       super(GaussianModel,self).__init__()
       self.input=prenet.input
       self.prenet=prenet
       self.middle=[]
       self.params=[]
       self.height=height
       self.width=width
       self.ballnum=ballnum
       self.sigma=sigma
       self.batchsize=batchsize

    def get_output(self, train=False):
        
        #self.middle=apply_model(self.nownet,X)
        self.middle=self.prenet.get_output(train)        
        
        #outball=apply_model(self.nownet,X)    
        #outball=self.middle
        ##########Main Function
        #Read Data        
        outmap=[]
        a=np.zeros((self.ballnum),dtype="int32")
        b=np.zeros((self.ballnum),dtype="int32")                        
        for batch in range(0,self.batchsize):
           for i in range(0,self.ballnum):
              a[i]=i
              b[i]=batch
           ballshare=theano.shared(np.array(a).T)           
           batchshare=theano.shared(np.array(b).T)
           results, updates = theano.scan(fn=self._loopoverallball ,
                                                           sequences=[ballshare,batchshare])           
           print 'dim1',results.ndim
           results=results.max(axis=[0])
           print 'dim2',results.ndim
           print results
           print 'prepare'           
           print 'gen'
           print 'compute'
           #print outball                          
           outmap.append(results)           
        output=T.reshape(outmap,(self.batchsize,1,self.height,self.width))
        return output

    def _loopoverallball(self, ballid,batchid):        
        ox=self.middle[batchid][ballid*2]#.reshape((1,1))
        print "ox:",ox.ndim
        x=T.tile(ox,(self.height,self.width))
        oy=self.middle[batchid][ballid*2+1]#.reshape((1,1))
        y=T.tile(oy,(self.height,self.width))
        w=T.tile(T.arange(0,self.width),(self.height,)).reshape((self.height,self.width))
        h=T.tile(T.arange(0,self.height).reshape((self.height,1)),(1,self.width))
        cof=(T.pow(x-w,2)+T.pow(y-h,2))*(-1.0/self.sigma)        
        print T.exp(cof).ndim
        return T.exp(cof)
        #return 1.0/(1+T.exp(-((T.exp(cof))-10.0)))
 
#####Ball Model

class BallModel(Layer):
    def __init__(self,prenet,batchsize):
        super(BallModel,self).__init__()
        self.prenet=prenet
        self.input=prenet.input
        self.middle=[]
        self.params=[]
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
        
        self.middle=self.prenet.get_output()
        print self.middle
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
        for batch in range(0,self.batchsize):
            d=self.middle[batch].reshape((26,)) ##ndim=0

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
            outputans.append(ret)
        outputans=T.reshape(outputans,(self.batchsize,144))
        return outputans

###XYZ - XY
class ConvertToXY(Layer):
    def __init__(self,prenet,batchsize,height,width,ballnum):
        super(ConvertToXY,self).__init__()
        self.prenet=prenet
        self.input=prenet.input
        self.middle=[]
        self.params=[]
        self.batchsize=batchsize
        self.height=height
        self.width=width
        self.ballnum=ballnum

    def get_output(self,train=False):        
        #self.middle=apply_model(self.prenet,X)
        self.middle=self.prenet.get_output(train)
        
        outputans=[]
        for batch in range(0,self.batchsize):
            X=self.middle[batch]
            for k in range(0,self.ballnum):
                outputans.append((X[3*k]  /X[3*k+2]+0.5)*self.width)
                outputans.append((X[3*k+1]/X[3*k+2]+0.5)*self.height)
        outputans=T.reshape(outputans,(self.batchsize,96))
        return outputans
