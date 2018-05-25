import numpy

class Node:

    def set_output(self,O): self.O = O

class Linear(Node):
    
    def __init__(self,I,W,B):

        self.I = I; I.set_output(self)
        self.W = W; W.set_output(self)
        self.B = B; B.set_output(self)
        self.reset()

    def reset(self): self.o,self.di,self.dw,self.db = None,None,None,None

    def evaluate(self):
        if self.o is None: self.o = self.I.evaluate().dot(self.W.evaluate())+self.B.evaluate()
        return self.o

    def grad(self):
        if self.di is None: self.di = self.O.grad().dot(self.W.evaluate().T)
        return self.di

    def gradW(self):
        if self.dw is None: self.dw = numpy.dot(self.I.evaluate().T,self.O.grad()) / len(self.I.evaluate())
        return self.dw

    def gradB(self):
        if self.db is None: self.db = self.O.grad().mean(axis=0)
        return self.db
        
class Tanh(Node):
    
    def __init__(self,I): self.I = I; I.set_output(self); self.reset()

    def reset(self): self.o,self.di = None,None
        
    def evaluate(self):
        if self.o is None: self.o = numpy.tanh(self.I.evaluate())
        return self.o
    
    def grad(self):
        if self.di is None: self.di = self.O.grad()*(1-self.o**2)
        return self.di

class Input(Node):
    
    def __init__(self): self.reset()
    def reset(self): self.o = None
    def feed(self,data): self.o = data
    def evaluate(self): return self.o

class Output(Node):
    
    def __init__(self,I): self.I = I; I.set_output(self); self.reset()
    def reset(self): self.di = None
    def feed(self,error): self.di = error
    def evaluate(self): return self.I.evaluate()
    def grad(self): return self.di

class Weight(Node):
    
    def __init__(self,w): self.w = w
    def evaluate(self): return self.w
    def grad(self): return self.O.gradW()

class Bias(Node):
    
    def __init__(self,b): self.b = b
    def evaluate(self): return self.b
    def grad(self): return self.O.gradB()

