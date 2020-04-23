import pandas as pd


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


X_train=train.iloc[:,:-1].values
Y_train=train.iloc[:,-1].values
X_test=test.iloc[:,:-1].values
Y_test=test.iloc[:,-1].values


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=50,
                                metric='minkowski',
                                p=1)
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)


correct=0
for i in range(len(cm)):
    correct += cm[i][i]
acc = correct/len(Y_pred)
#accuracy of knn = 0.81947 (Euclidean)
#accuracy of knn = 0.89480 (Manhattan)

