import misc
import numpy as np
from PCA import pca
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
		#-------------------------------Aplicacao do PCA------------------------------
		Z = pca(dataset_instances,K)
		#---------------Particionando o dataset em 30% teste 70% treino------------------------------
		trainset_instances = Z[range(int(n*0.9)),:]
		testset_instances = Z[range(int(n*0.9),n),:]
		trainset_class = dataset_class[range(int(n*0.9))]
		testset_class = dataset_class[range(int(n*0.9),n)]
		
		#-------------------------------------------Verificacao de acuracia com 3-nn------------------------------------------------
		#treinamento
		knn = KNeighborsClassifier(n_neighbors=9)  
		knn.fit(trainset_instances , trainset_class)
		#score
		score = knn.score(testset_instances,testset_class) * 100
		print "Para "+repr(K)+" Autovetor(es)"+" Acuracia media 3-nn: "+repr(score) + '%'
main()
