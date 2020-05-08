# This is Campus Recruitment Dataset from Kaggle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

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

#-------------Applying Artificial Neural Network to the dataset using PyTorch-------------

class ANN(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(ANN, self).__init__()
        self.h1 = nn.Linear(inputsize, 20)
        self.h2 = nn.Linear(20, 20)
        self.h3 = nn.Linear(20, 20)
        self.h4 = nn.Linear(20, outputsize)
        self.activation = nn.ReLU()
        self.activation1 = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.h1(x))
        x = self.activation(self.h2(x))
        x = self.activation(self.h3(x))
        output = self.activation1(self.h4(x))
        return output

learningRate = 0.001
epochs = 180

classifier = ANN(inputsize = 14, outputsize = 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr = learningRate)

X_train = np.array(X_train, dtype = np.float32)
y_train = np.array(y_train, dtype = np.float32)
X_test = np.array(X_test, dtype = np.float32)


for epoch in range(1, epochs+1):
    inputs = Variable(torch.from_numpy(X_train))
    targets = Variable(torch.from_numpy(y_train))
    optimizer.zero_grad()
    outputs = classifier(inputs)
    loss = criterion(outputs, targets.squeeze())
    print(loss)
    loss.backward()
    optimizer.step()
    print('Epoch = ', format(epoch), 'loss = ', format(loss))

y_pred = classifier(Variable(torch.from_numpy(X_test))).data.numpy()
y_pred = torch.round(Variable(torch.from_numpy(y_pred))).data.numpy()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]))*100

y_train_pred = classifier(Variable(torch.from_numpy(X_train))).data.numpy()
y_train_pred = torch.round(Variable(torch.from_numpy(y_train_pred))).data.numpy()

from sklearn.metrics import accuracy_score
Accuracy_of_trainingset = accuracy_score(y_train_pred, y_train)*100











