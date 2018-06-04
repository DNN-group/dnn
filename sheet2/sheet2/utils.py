import numpy, numpy.random

import sklearn,sklearn.datasets

def getBostonHousingData():
    
    # Extract and preprocess data
    D = sklearn.datasets.load_boston()
    X = D['data']
    T = D['target']

    # Partition the data
    N = len(X)
    perm = numpy.random.mtrand.RandomState(1).permutation(N)
    Xtrain,Xtest = X[perm[:N//2]],X[perm[N//2:]]
    Ttrain,Ttest = T[perm[:N//2]],T[perm[N//2:]]

    # Normalize input data
    m,s = Xtrain.mean(axis=0),Xtrain.std(axis=0)
    for x in Xtrain,Xtest: x -= m; x /= s

    # Normalize targets
    m,s = Ttrain.mean(),Ttrain.std()
    for t in Ttrain,Ttest: t -= m; t /= s

    return Xtrain,Ttrain,Xtest,Ttest

