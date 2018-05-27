import misc
import numpy as np
from PCA import pca
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.decomposition import PCA

	#plt.scatter(Z[:,0],Z[:,1])
	#plt.title("2d Dataset", fontsize='large')
	#plt.show()
def main():
	dataset = misc.loadData('cm1.arff')
	dataset = np.asarray(dataset)
	np.random.shuffle(dataset)
	dataset_instances = dataset[:,range(len(dataset[0])-1)]
	dataset_class = dataset[:,-1]
	n = dataset_instances.shape[0]

	for K in range(1,dataset_instances.shape[1]+1):

		#-------------------------------PCA application-------------------------------
		Z = pca(dataset_instances,K)
		#-------------------------------cross validation------------------------------
		knn = KNeighborsClassifier(n_neighbors=3) 
		score = cross_val_score(knn, Z, dataset_class, cv=10)

		print "For "+repr(K)+" Eigenvectors: "+" Accuracy using 3-nn = "+repr(score.mean()*100) + '%'

main()
