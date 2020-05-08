# This is Campus Recruitment Dataset from Kaggle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Placement_Data_Full_Class.csv')

X = dataset.iloc[:, 1:13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])
X[:, 2] = le_X.fit_transform(X[:, 2])
X[:, 4] = le_X.fit_transform(X[:, 4])
X[:, 5] = le_X.fit_transform(X[:, 5])
X[:, 7] = le_X.fit_transform(X[:, 7])
X[:, 8] = le_X.fit_transform(X[:, 8])
X[:, 10] = le_X.fit_transform(X[:, 10])

X5 = X[:, 5].reshape(-1, 1)
X7 = X[:, 7].reshape(-1, 1)

ohe = OneHotEncoder()
X5 = ohe.fit_transform(X5).toarray()
X5 = X5[:, 1:3]

X7 = ohe.fit_transform(X7).toarray()
X7 = X7[:, 1:3]

X_temp1 = X[:, 0:5]
X_temp2 = X[:, 6]
X_temp2 = X_temp2.reshape(-1, 1)
X_temp3 = X[:, 8:12]

X = np.concatenate((X_temp1, X5, X_temp2, X7, X_temp3), axis = 1)

y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
y = le_y.fit_transform(y)
y = y.reshape(-1, 1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#---------Applying SVM--------------------------------------------------------

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]))*100

y_train_pred = classifier.predict(X_train)

from sklearn.metrics import accuracy_score
Accuracy_of_trainingset = accuracy_score(y_train_pred, y_train)*100





















