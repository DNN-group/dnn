import numpy, numpy.random

def getdata(N):
    rstate = numpy.random.mtrand.RandomState(123)
    X = rstate.normal(0,1,[N,2])
    T = rstate.normal(0,1,[N,1])
    return X,T

def getparameters(sizes):
    rstate = numpy.random.mtrand.RandomState(234)
    W = [rstate.normal(0,m**-.5,[m,n]) for m,n in zip(sizes[:-1],sizes[1:])]
    B = [numpy.zeros([n]) for n in sizes[1:]]
    return W,B

