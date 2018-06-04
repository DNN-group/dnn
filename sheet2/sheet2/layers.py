import numpy

class Layer:

    def pgradstep(self,lr): pass
    def computepgrad(self): pass

class Sequential:
    
    def __init__(self,layers): self.layers = layers
        
    def forward(self,Q):
        for l in self.layers: Q = l.forward(Q)
        return Q

    def backward(self,DQ):
        for l in self.layers[::-1]: DQ = l.backward(DQ)
        return DQ

class Linear(Layer):
    
    def __init__(self,m,n,seed=0):
        self.W = numpy.random.mtrand.RandomState(seed).normal(0,m**-.5,[m,n])
        self.B = numpy.zeros([n])
        self.DWm,self.DBm = 0,0
        
    def forward(self,A):
        self.A = A*1; return A.dot(self.W)+self.B
    
    def backward(self,DZ):
        self.DZ = DZ; return DZ.dot(self.W.T)

    def computepgrad(self):
        self.DW = numpy.dot(self.A.T,self.DZ)/len(self.A)
        self.DB = self.DZ.mean(axis=0)

    def pgradstep(self,lr):
        self.DWm = 0.99*self.DWm+self.DW; self.W -= lr*self.DWm
        self.DBm = 0.99*self.DBm+self.DB; self.B -= lr*self.DBm

class ReLU(Layer):

    def forward(self,Z):
        self.Z = Z; return numpy.maximum(0,self.Z)

    def backward(self,DA):
        return DA*(self.Z>0)

class Pooling(Layer):

    def __init__(self,n):
        self.V = numpy.ones([n])*n**-.5; self.V[::2] *= -1

    def forward(self,Z):
        return (self.V*Z).sum(axis=1)

    def backward(self,DA):
        return self.V*DA[:,numpy.newaxis]

