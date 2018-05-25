import numpy


class Sequential:
    
    def __init__(self,layers): self.layers = layers
        
    def forward(self,Q):
        for l in self.layers: Q = l.forward(Q)
        return Q

    def backward(self,DQ):
        for l in self.layers[::-1]: DQ = l.backward(DQ)
        return DQ
    
class Linear:
    
    def __init__(self,W,B): self.W,self.B = W,B
        
    def forward(self,A):
        self.A = A*1;
        return A.dot(self.W)+self.B
    
    def backward(self,DZ):
        self.DW = numpy.dot(self.A.T,DZ)/len(self.A)
        self.DB = DZ.mean(axis=0)
        return DZ.dot(self.W.T)

class Tanh:
    
    def forward(self,Z): self.A = numpy.tanh(Z); return self.A
    def backward(self,DA): return DA*(1-self.A**2)

