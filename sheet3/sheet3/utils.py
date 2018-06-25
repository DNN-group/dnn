import matplotlib
from matplotlib import pyplot as plt
import numpy

def images(X):
    X = X - X.min(); X = X / X.max()
    X = X.reshape([5,10,32,32,3])
    X = numpy.pad(X,((0,0),(0,0),(4,4),(4,4),(0,0)),'constant', constant_values=(1,1))
    X = X.transpose([0,2,1,3,4]).reshape([5*40,10*40,3])
    plt.figure(figsize=(10,5))
    plt.axis('off')
    plt.imshow(X,cmap='seismic',vmin=-1,vmax=1)

def heatmaps(H):
    H = H / H.max()
    H = H.reshape([5,10,32,32])
    H = numpy.pad(H,((0,0),(0,0),(4,4),(4,4)),'constant', constant_values=(0,0))
    H = H.transpose([0,2,1,3]).reshape([5*40,10*40])
    plt.figure(figsize=(10,5))
    plt.axis('off')
    plt.imshow(H,cmap='seismic',vmin=-1,vmax=1)


