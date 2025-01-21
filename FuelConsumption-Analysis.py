import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import PolynomialFeatures

"""
data = pd.read_csv("FuelConsumption.csv")
x_1 = data.iloc[:,4:5].values
x_2 = data.iloc[:,11:12].values
y = data.iloc[:,-1]
plt.scatter(x_1, y)
plt.show()
plt.scatte
"""
#2
data = pd.read_csv("FuelConsumption.csv")

x1 = data.iloc[:,4:5]
x2=data.iloc[:,11:12]
y= data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x1, y, shuffle=True, test_size=0.15, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print("MSE Eginsize", mean_squared_error(y_test, y_hat))
print("r2 Enginsize", r2_score(y_test, y_hat))

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_hat)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x2, y, shuffle=True, test_size=0.15, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print("MSE x2", mean_squared_error(y_test, y_hat))
print("r2 x2", r2_score(y_test, y_hat))

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_hat)
plt.show()


#👏 This code belongs to Queen Ahlam! 👑
#DO YOU LIKE IT? let me know my flower :)
