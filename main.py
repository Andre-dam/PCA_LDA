import misc
import numpy as np
from PCA import pca

def main():
    dataset = misc.loadData('sample.csv')
    print dataset
    dataset = np.asarray(dataset)
    #print dataset
    Z = pca(dataset)
    #print Z
main()