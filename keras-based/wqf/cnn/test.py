import theano
import theano.tensor as T
import numpy as np
a=theano.shared(np.zeros((3,3),dtype='float32'))
print a.eval()
a=T.set_subtensor(a[0,0],0.3)
a=T.set_subtensor(a[0,1],-0.4)
a=T.set_subtensor(a[0,2],0.5)
a=T.set_subtensor(a[1,0],1.3)
a=T.set_subtensor(a[1,1],-0.9)
a=T.set_subtensor(a[1,2],0.7)
a=T.set_subtensor(a[2,0],1.6)
a=T.set_subtensor(a[2,1],3.4)
a=T.set_subtensor(a[2,2],-9.0)
print a.eval()
batch=3
h=3
w=4
b=a[:,2].reshape((batch,1))
print b.eval()
c=T.tile(b,(1,h*w)).reshape((batch,h,w))
print "origin:",c.eval()

multiply=a[:,1].reshape((batch,1))
multiply=T.tile(multiply,(1,h*w)).reshape((batch,h,w))
print multiply.ndim
print multiply.eval()


c=multiply*T.exp(-T.pow(c,2))
print "exp:",c.eval()
print c.ndim
print c

print c.eval()

e=theano.shared(np.zeros((3,4),dtype='float32'))
e=T.set_subtensor(e[:,2],3.0)
print e.eval()

w1=T.tile(T.tile(T.arange(0,w),(h,)),(batch,)).reshape((batch,h,w))
print w1.eval()

m1=theano.shared(np.zeros((3,3,3),dtype='float32'))
m2=theano.shared(np.zeros((3,3,3),dtype='float32'))
m1=T.set_subtensor(m1[0,0,0],2.0)
m1=T.set_subtensor(m1[0,0,1],-3.0)
m1=T.set_subtensor(m1[0,0,2],4.0)
m1=T.set_subtensor(m1[0,1,0],12.0)
m1=T.set_subtensor(m1[0,1,1],5.0)
m1=T.set_subtensor(m1[0,1,2],7.0)
m1=T.set_subtensor(m1[0,2,0],-6.0)
m1=T.set_subtensor(m1[0,2,1],9.0)
m1=T.set_subtensor(m1[0,2,2],10.0)

m1=T.set_subtensor(m1[1,0,0],12.0)
m1=T.set_subtensor(m1[1,0,1],6.0)
m1=T.set_subtensor(m1[1,0,2],8.0)
m1=T.set_subtensor(m1[1,1,0],5.0)
m1=T.set_subtensor(m1[1,1,1],9.0)
m1=T.set_subtensor(m1[1,1,2],17.0)
m1=T.set_subtensor(m1[1,2,0],23.0)
m1=T.set_subtensor(m1[1,2,1],49.0)
m1=T.set_subtensor(m1[1,2,2],-50.0)

m1=T.set_subtensor(m1[2,0,0],62.0)
m1=T.set_subtensor(m1[2,0,1],-73.0)
m1=T.set_subtensor(m1[2,0,2],14.0)
m1=T.set_subtensor(m1[2,1,0],2.0)
m1=T.set_subtensor(m1[2,1,1],96.0)
m1=T.set_subtensor(m1[2,1,2],27.0)
m1=T.set_subtensor(m1[2,2,0],-36.0)
m1=T.set_subtensor(m1[2,2,1],99.0)
m1=T.set_subtensor(m1[2,2,2],67.0)

##---------------
m2=T.set_subtensor(m2[0,0,0],4.0)
m2=T.set_subtensor(m2[0,0,1],-11.0)
m2=T.set_subtensor(m2[0,0,2],6.0)
m2=T.set_subtensor(m2[0,1,0],8.0)
m2=T.set_subtensor(m2[0,1,1],25.0)
m2=T.set_subtensor(m2[0,1,2],17.0)
m2=T.set_subtensor(m2[0,2,0],-3.0)
m2=T.set_subtensor(m2[0,2,1],-9.0)
m2=T.set_subtensor(m2[0,2,2],20.0)

m2=T.set_subtensor(m2[1,0,0],3.0)
m2=T.set_subtensor(m2[1,0,1],4.0)
m2=T.set_subtensor(m2[1,0,2],2.0)
m2=T.set_subtensor(m2[1,1,0],1.0)
m2=T.set_subtensor(m2[1,1,1],5.0)
m2=T.set_subtensor(m2[1,1,2],7.0)
m2=T.set_subtensor(m2[1,2,0],78.0)
m2=T.set_subtensor(m2[1,2,1],59.0)
m2=T.set_subtensor(m2[1,2,2],-20.0)

m2=T.set_subtensor(m2[2,0,0],-42.0)
m2=T.set_subtensor(m2[2,0,1],53.0)
m2=T.set_subtensor(m2[2,0,2],24.0)
m2=T.set_subtensor(m2[2,1,0],32.0)
m2=T.set_subtensor(m2[2,1,1],16.0)
m2=T.set_subtensor(m2[2,1,2],-17.0)
m2=T.set_subtensor(m2[2,2,0],46.0)
m2=T.set_subtensor(m2[2,2,1],29.0)
m2=T.set_subtensor(m2[2,2,2],87.0)
print 'm1:'
print m1.eval()
print '------------------------------------'
print 'm2:'
print m2.eval()
m3=m1+m2
print 'm1+m2:'
print m3.eval()
b=3
h=3
w=3
minx=T.tile(m3.min(axis=[1,2]).reshape((b,1)),(1,h*w)).reshape((b,(h*w))).reshape((b,h,w))
print 'minx:'
print minx.eval()
maxx=T.tile(m3.max(axis=[1,2]).reshape((b,1)),(1,h*w)).reshape((b,(h*w))).reshape((b,h,w))
print 'maxx:'
print maxx.eval()
print 'normalization:'
results=(m3-minx)/(maxx-minx)
print results.eval()

print 'h1:'
h1=T.tile(T.tile(T.arange(0,h).reshape((h,1)),(1,w)),(b,1)).reshape((b,h,w))
print h1.ndim
print h1.eval()