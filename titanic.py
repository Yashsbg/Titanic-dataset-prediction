# dependencies
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

data_test = pd.read_csv("test(1).csv")
data_train = pd.read_csv("train(1).csv")

data_train = data_train.drop(['PassengerId'], axis=1)

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [data_train, data_test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
data_train = data_train.drop(['Cabin'], axis=1)
data_test = data_test.drop(['Cabin'], axis=1)

data = [data_train, data_test]
for dataset in data:
    mean = data_train["Age"].mean()
    std = data_test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean-std, mean+std, size=is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = data_train["Age"].astype(int)
data_train.isnull().sum()

data_train['Embarked'] = data_train['Embarked'].fillna('S')

data = [data_train, data_test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

data_train = data_train.drop('Name', axis=1)
data_test = data_test.drop('Name', axis=1)

genders = {"male": 0,"female": 1}
data = [data_train, data_test]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

data_train = data_train.drop("Ticket", axis=1)
data_test = data_test.drop("Ticket", axis=1)

ports = {"S":0, "C":1, "Q":2}
data = [data_test, data_train]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

data = [data_train, data_test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

data = [data_train, data_test]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

x_train = data_train.drop("Survived", axis=1)
y_train = data_train["Survived"]
x_test = data_test.drop("PassengerId", axis=1).copy()


logReg = LogisticRegression()
logReg.fit(x_train, y_train)
y_pred = logReg.predict(x_test)
acc_log = round(logReg.score(x_train, y_train) * 100, 2)


DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)
y_pred = DTC.predict(x_test)
acc_log = round(DTC.score(x_train, y_train) * 100, 2)


svc = LinearSVC(max_iter=5000)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_log = round(svc.score(x_train, y_train) * 100, 2)


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_log = round(knn.score(x_train, y_train) * 100, 2)
acc_log