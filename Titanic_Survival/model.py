import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Getting the dataset
df = pd.read_csv("train.csv")
df.dropna(inplace=True)

X_train = df.iloc[:, [2,4,5,6,7,9]]
y_train = df.iloc[:, 1]

df_test = pd.read_csv("test.csv")
df_test.dropna(inplace=True)

X_test = df_test.iloc[:, [1,3,4,5,6,8]]

#Handling categorical value
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X_train.iloc[:, 1] = le.fit_transform(X_train.iloc[:, 1])
ohe = OneHotEncoder(categorical_features=[1])
X_train = ohe.fit_transform(X_train).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X_test.iloc[:, 1] = le.fit_transform(X_test.iloc[:, 1])
ohe = OneHotEncoder(categorical_features=[1])
X_test = ohe.fit_transform(X_test).toarray()

#training and test spliting in train.csv tftr is test from train
from sklearn.cross_validation import train_test_split
X_train, X_tftr, y_train, y_tftr = train_test_split(X_train, y_train)

#Training on SVM
from sklearn.svm import SVC
regressor = SVC(kernel='rbf')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_tftr)

#final predicting using test.csv
y_test = regressor.predict(X_test)

from sklearn.metrics import make_scorer, accuracy_score
print(accuracy_score(y_tftr, y_pred)) #Accuracy with SVM 65.21