
# coding: utf-8

# # PCA and LDA

# This report aims to evaluate the implemented PCA and LDA algorithms. In evaluation, two databases of the Promise repository will be used: http://promise.site.uottawa.ca/SERepository/datasets-page.html

# # PCA

# **Principal Component Analysis (PCA)** is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the large set.
# 
# The main idea of principal component analysis (PCA) is to reduce the dimensionality of a data set consisting of many variables correlated with each other, either heavily or lightly, while retaining the variation present in the dataset, up to the maximum extent. The same is done by transforming the variables to a new set of variables, which are known as the principal components (or simply, the PCs) and are orthogonal, ordered such that the retention of variation present in the original variables decreases as we move down in the order. So, in this way, the 1st principal component retains maximum variation that was present in the original components. The principal components are the eigenvectors of a covariance matrix, and hence they are orthogonal.
# 

# In[108]:


#Used Modules
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#Developed Modules
import misc
import PCA


# ### Introduction

# Here, our desired outcome of the principal component analysis is to project a feature space (our dataset consisting of *n* *m*-dimensional samples) onto a smaller subspace that represents our data “well”.
# 
# **The basic steps for PCA algorithm are:**
# 
# - The mean is calculated and the entire data set normalized.
# - The covariance matrix is calculated.
# - The eigenvectors and eigenvalues of the covariance matrix are then calculated.
# - The K eigenvectors with the greatest amount of associated information are chosen.
# - We assemble the projection matrix P based on previously selected eigenvectors.
# - The normalized image obtained in step 1 is projected by the projection matrix.
# 
# 
# 

# In[2]:


#---Using an artificial dataset with only two dimensions to illustrate how the PCA works.
dataset = misc.loadData('sample.csv')
dataset = np.asarray(dataset)

plt.scatter(dataset[:,0],dataset[:,1])
plt.title("2D Sample - Dataset", fontsize='large')
plt.show()


# The number of eigenvectors generated is equal to the number of dimensions of the dataset, so in the present example, two eigenvectors are gerenerated, each with the same dimension as the dataset(two dimensions).  

# In[12]:


#---The two eigenvectors are:
eigenValues, eigenVectors = PCA.pca_eigen(dataset)
np.set_printoptions(precision=3)
print eigenVectors[:,0],eigenVectors[:,1]


# In[8]:


#---The center of the distribution is the mean of each coordinate:
x_center = sum(dataset[:,0])/dataset.shape[0]
y_center = sum(dataset[:,1])/dataset.shape[0]

#---Representing orientation of the vectors in 2-Dimensions:
ev_x = eigenVectors[0,:]
ev_y = eigenVectors[1,:]


plt.quiver([x_center, x_center], [y_center, y_center], ev_x, ev_y, angles='xy', scale_units='xy', scale=0.01)
plt.scatter(dataset[:,0],dataset[:,1])
plt.title("2D Sample - Dataset Eigenvectors", fontsize='large')
plt.show()


# The Eigenvalues associated for each Eigenvectors will be used to know how much of information it's Eigenvector carries. 

# In[11]:


#---The eigenvalues are:
print eigenValues
#---Where the percent of information distribuition for each vector is:

plt.bar(['eval1: '+str(int(eigenValues[0])),'eval2: '+str(int(eigenValues[1]))],[100*eigenValues[0]/sum(eigenValues),100*eigenValues[1]/sum(eigenValues)])
plt.ylabel('%')
plt.show()


# As observeved on the previous chart, around 99% of all information is representend on the first eigenvector, thus, the dataset can be redimensioned to one dimension using just this vector, without significant loss of information. 

# In[31]:


#---Applying PCA using just the first eigenvector
print "Dataset before PCA dimension: "+repr(dataset.shape)
Z = PCA.pca(dataset,1)
print "Dataset after PCA dimension: "+repr(Z.shape)
plt.plot(Z, np.zeros(35), 'o')
plt.title("1D Sample - Dataset", fontsize='large')
plt.show()


# ## Results

# #### Using the CM1 dataset from promise repository http://promise.site.uottawa.ca/SERepository/datasets/cm1.arff

# In[132]:


#---loading dataset
dataset = misc.loadData('cm1.arff')
dataset = np.asarray(dataset)

#---spliting it in instances and class
dataset_attributes = dataset[:,range(len(dataset[0])-1)]
dataset_class = dataset[:,-1]

eigenValues, eigenVectors = PCA.pca_eigen(dataset_attributes)
print "The Dataset contains: "+repr(dataset.shape[0])+" instances and "+repr(dataset_attributes.shape[1])+" feature dimensions"


# In[57]:


#---The eigenvalues are:
print "Eigenvalues:\n",eigenValues

#---Where the percent of information distribuition for each vector is:
x = []
for i in range(dataset_attributes.shape[1]):
    x.append(str(i+1))
y = []
print "\nDescent ordered eigenvalues information distribution:"
for i in range(dataset_attributes.shape[1]):
    y.append(100*eigenValues[i]/sum(eigenValues))
    print str(y[i])+' %'

plt.bar(x,y)
plt.xlabel('Descent ordered eigenvalues')
plt.ylabel('%')
plt.show()


# In[93]:


#---In order to verify the influence of each vector and it's capability to represent the data, a comparison
#is then made, where the K(number of cumulative descending eigenvectors) will vary from 1 to the max number
#of dimensions present in the data, and a 1-nn classifier will be user to verify the influence of each K.

scores_ = []
k_ = []
for K in range(1,dataset_attributes.shape[1]+1):
    k_.append(K)
    #-------------------------------PCA application---------
    Z = PCA.pca(dataset_attributes,K)
    #-------------------------------cross validation--------
    knn = KNeighborsClassifier(n_neighbors=1) 
    score = cross_val_score(knn, Z, dataset_class, cv=4)
    scores_.append(score.mean()*100)
    print "For "+repr(K)+" Eigenvectors: "+" Accuracy using 3-nn = "+repr(score.mean()*100) + '%'
    
plt.plot(k_,scores_,0,100)
plt.title("Number of Eigenvectos x Accuracy", fontsize='large')
plt.ylim(0,100)
plt.show()
plt.plot(k_,scores_)
plt.title("Number of Eigenvectos x Accuracy zoomed", fontsize='large')
plt.show()


# **Time analisys**

# In[139]:


#---A time analisys is then made in order to compare how a dimension reduction using PCA
#might improve some algorithms running time, a 10-nn classifier will be used for that test.

#First it will run a 10-fold cross validation on raw dataset, that is, without using PCA
start = time.time()
knn = KNeighborsClassifier(n_neighbors=10)
score = cross_val_score(knn, dataset_attributes, dataset_class, cv=10)
end = time.time()
print "Elapsed time without PCA: "+repr(int((end - start)*1000))+" ms"

#Then it will run a 10-fold cross validation on 1D dataset, that is, using PCA with K=1
Z = PCA.pca(dataset_attributes,1)

start = time.time()
knn = KNeighborsClassifier(n_neighbors=10)
score = cross_val_score(knn, Z, dataset_class, cv=10)
end = time.time()
print "Elapsed time with PCA: "+repr(int((end - start)*1000))+" ms"


# #### Using the JM1 dataset from promise repository http://promise.site.uottawa.ca/SERepository/datasets/cm1.arff

# In[141]:


#---loading dataset
dataset = misc.loadData('kc1.arff')
dataset = np.asarray(dataset)

#---spliting it in instances and class
dataset_attributes = dataset[:,range(len(dataset[0])-1)]
dataset_class = dataset[:,-1]

eigenValues, eigenVectors = PCA.pca_eigen(dataset_attributes)
print "The Dataset contains: "+repr(dataset.shape[0])+" instances and "+repr(dataset_attributes.shape[1])+" feature dimensions"


# In[100]:


#---The eigenvalues are:
print "Eigenvalues:\n",eigenValues

#---Where the percent of information distribuition for each vector is:
x = []
for i in range(dataset_attributes.shape[1]):
    x.append(str(i+1))
y = []
print "\nDescent ordered eigenvalues information distribution:"
for i in range(dataset_attributes.shape[1]):
    y.append(100*eigenValues[i]/sum(eigenValues))
    print str(y[i])+' %'

plt.bar(x,y)
plt.xlabel('Descent ordered eigenvalues')
plt.ylabel('%')
plt.show()


# In[107]:


#---In order to verify the influence of each vector and it's capability to represent the data, a comparison
#is then made, where the K(number of cumulative descending eigenvectors) will vary from 1 to the max number
#of dimensions present in the data, and a 3-nn classifier will be user to verify the influence of each K.

scores_ = []
k_ = []

for K in range(1,dataset_attributes.shape[1]+1):
    k_.append(K)
    #-------------------------------PCA application---------
    Z = PCA.pca(dataset_attributes,K)
    #-------------------------------cross validation--------
    knn = KNeighborsClassifier(n_neighbors=3) 
    score = cross_val_score(knn, Z, dataset_class, cv=10)
    scores_.append(score.mean()*100)
    print "For "+repr(K)+" Eigenvectors: "+" Accuracy using 3-nn = "+repr(score.mean()*100) + '%'
    
plt.plot(k_,scores_,0,100)
plt.title("Number of Eigenvectos x Accuracy", fontsize='large')
plt.ylim(0,100)
plt.show()
plt.plot(k_,scores_)
plt.title("Number of Eigenvectos x Accuracy (zoomed)", fontsize='large')
plt.show()


# **Time analisys**

# In[142]:


#---A time analisys is then made in order to compare how a dimension reduction using PCA
#might improve some algorithms running time, a 10-nn classifier will be used for that test.

#First it will run a 10-fold cross validation on raw dataset, that is, without using PCA
start = time.time()
knn = KNeighborsClassifier(n_neighbors=10)
score = cross_val_score(knn, dataset_attributes, dataset_class, cv=10)
end = time.time()
print "Elapsed time without PCA: "+repr(int((end - start)*1000))+" ms"

#Then it will run a 10-fold cross validation on 1D dataset, that is, using PCA with K=1
Z = PCA.pca(dataset_attributes,1)

start = time.time()
knn = KNeighborsClassifier(n_neighbors=10)
score = cross_val_score(knn, Z, dataset_class, cv=10)
end = time.time()
print "Elapsed time with PCA: "+repr(int((end - start)*1000))+" ms"


# ## Conclusions

# The values presented in Table 1 below demonstrate the expected, increasing the number of eigenvectors leads the rate of accuracy to increase as well. Another observation to be made is that the smaller the difference between the accuracys as the amount of eigenvectors(K) increses, the better is, this means that the dataset can be well represented with a reduced size, thereby increasing performance and decreasing processing time for some learning algorithm that turns out to be used in the future. This fact is visible in table 2 where it is possible to verify this characteristic, it is also visible that the larger the dataset the better the improvement in the reduction of computation time made by the reduction of size.

# **Table - Accuracy for diferent k values**
# 
# |    | k=1 |k=10| k=21 |
# |--------|--------|--------|------|
# |   CM1 | 84.93 %|  85.33 % | 85.33 % |
# |   KC1 | 80.09 %|80.89 %| 80.94 % |
# 
# 
# **Table - Time elapsed in ms**
# 
# | This   | no-PCA | PCA K=1 | 
# |--------|--------|--------|------|
# |   CM1 (498 instances) |45ms| 36ms | 
# |   KC1 (2109 instances) | 112ms | 94ms  |
