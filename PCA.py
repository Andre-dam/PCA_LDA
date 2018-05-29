from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pca(X,K):
    X=X.astype('float64')
    #------------------mean removal of each feature-----------------------
    X_norm = preprocessing.scale(X, with_std = False)

    #----------------------covariance matrix------------------------------
    X_cov = 1.0/len(X_norm) * np.matmul(X_norm.T , X_norm)
    
    #------------eigenvalues and eigenvectors computation-----------------
    eigenValues, eigenVectors = np.linalg.eig(X_cov)

    #-----------sort eigenvalues and eigenvectors descending--------------
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    #------------------------dimension change-----------------------------
    Z = np.matmul(X_norm,eigenVectors[:,range(K)])

    return Z

def pca_eigen(X):
    X=X.astype('float64')
    #------------------mean removal of each feature-----------------------
    X_norm = preprocessing.scale(X, with_std = False)

    #----------------------covariance matrix------------------------------    
    X_cov = np.cov(X_norm.T)
    #------------eigenvalues and eigenvectors computation-----------------
    eigenValues, eigenVectors = np.linalg.eig(X_cov)

    #-----------sort eigenvalues and eigenvectors descending--------------
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues, eigenVectors