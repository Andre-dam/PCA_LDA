from sklearn import preprocessing
import numpy as np
def pca(X):
    #mean removal of each feature
    X_norm = preprocessing.scale(X, with_std = False)
    #covariance matrix
    X_cov = np.cov(X_norm.transpose())
    eigenValues, eigenVectors = np.linalg.eig(X_cov)
    print eigenValues, eigenVectors

    #sort eigenvalues and eigenvectors descending
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    print eigenValues, eigenVectors