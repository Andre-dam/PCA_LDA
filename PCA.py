from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def pca(X,K):
    #mean removal of each feature
    X=X.astype('float64')
    X_norm = preprocessing.scale(X, with_std = False)
    #covariance matrix a transposicao pois linha= variavel coluna = observacao
    X_cov = 1.0/len(X_norm) * np.matmul(X_norm.T , X_norm)

    eigenValues, eigenVectors = np.linalg.eig(X_cov)

    #sort eigenvalues and eigenvectors descending
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    #print eigenValues
    Z = np.matmul(X_norm,eigenVectors[:,range(K)])

    
    return Z