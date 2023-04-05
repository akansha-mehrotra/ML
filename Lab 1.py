from sklearn.datasets import load_iris
iris=load_iris()
iris

type(iris)
print(iris.target_names,iris.feature_names)
iris.keys()

print(iris["target_names"])
print(iris.target_names)

n_samples,n_features=iris.data.shape
print("Number of samples",n_samples)
print("Number of features",n_features)

print(iris.data[0])
iris.data[[12,26,89,114]]

print(iris.data.shape)
print(iris.target.shape)
print(iris.target)
import numpy as np
np.bincount(iris.target)
print(iris.target_names)
print(iris.target)

#sklearn contains a "wine data set"
''' 
Find and load this dataset
Can you find a description?
What are the name of the classes?
What are the features?
Count for each feature and class
'''

from sklearn.datasets import load_wine
wine=load_wine()
wine

type(wine)
wine.keys()

import numpy as np
np.bincount(wine.target)

n_samples,n_features=wine.data.shape
print("Number of class",n_samples)
print("Number of features",n_features)

print(wine.feature_names)
