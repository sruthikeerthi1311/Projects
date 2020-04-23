import pandas as pd


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


X_train=train.iloc[:,:-1].values
Y_train=train.iloc[:,-1].values
X_test=test.iloc[:,:-1].values
Y_test=test.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)


from keras.utils import to_categorical
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)


from keras.models import Sequential
from keras.layers import Dense


model=Sequential()
model.add(Dense(output_dim=281,activation='relu',input_dim=562))
model.add(Dense(output_dim=141,activation='relu'))
model.add(Dense(output_dim=71,activation='relu'))
model.add(Dense(output_dim=36,activation='relu'))
model.add(Dense(output_dim=18,activation='relu'))
model.add(Dense(output_dim=9,activation='relu'))
model.add(Dense(output_dim=6,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=50)


predict=model.predict_classes(X_test)


scores=model.evaluate(X_test,Y_test)
print("\n%s : %2f%%" %(model.metrics_names[1],scores[1]*100))