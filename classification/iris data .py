from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris=datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data)
type(iris.target)
print(iris.data.shape)
print(iris.target_names)
X=iris.data
y=iris.target
data=pd.DataFrame(X,columns=iris.feature_names)
print(data.head())
pd.plotting.scatter_matrix(data,c=y,figsize=[8,8],s=180,marker='D')
data.info()
data.describe()


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X,y)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.3, random_state = 21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred
y_test
