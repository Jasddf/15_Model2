import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 데이터 전처리
data = pd.read_csv('./data/1.salary.csv')
array = data.values
array.shape
x = array[:, 0] #독립변수(종속변수에 영향을 줄 수 있는 변수)
y = array[:, 1] #종속변수(독립변수에 따라서 바뀔 수 있는 변수)
# 근속연수 = 연봉
x.reshape(-1,1)

# 데이터 시각화 자료
# plt.clf()
# plt.scatter(x,y, label='data', color='blue', marker='*', s=30, alpha=0.5)
# plt.title('Scatter chart')
# plt.xlabel('Experience Years')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()

(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

plt.scatter(x,y, label='data', color='blue', marker='*', s=30, alpha=0.5)
plt.clf()
plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='x')

plt.title('Salary chart')
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.legend()
plt.show()

mean = mean_absolute_error(Y_test, y_pred)
print(mean)

