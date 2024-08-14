import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 데이터 전처리
header = ['sepal-length','sepal-width','petal-length','petal-width','class']
data = pd.read_csv('./data/2.iris.csv',names=header)
array = data.values
array.shape
X = array[:,0:4]
Y = array[:,4]


# 데이터 분할
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3)
model = DecisionTreeClassifier()
y_pred = model.predict(X_test)

fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, X, Y, cv=fold, scoring='accuracy')
s = sum(acc)
l = len(acc)


plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='x')

plt.title('Salary chart')
plt.xlabel('Sepal,Petal')
plt.ylabel('Flower Class')
plt.legend()
plt.show()


