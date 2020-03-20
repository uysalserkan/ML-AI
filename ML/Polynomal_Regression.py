import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

""" 
 * Polynomial Linear Regression
 ! y = b0 + b1*x + b2*x^2 + ...
"""
df = pd.read_csv("polynomial.csv", sep=";")

y = df.araba_max_hiz.values.reshape(-1, 1)
x = df.araba_fiyat.values.reshape(-1, 1)

polynomial_reg = PolynomialFeatures(degree=3)

xPower2 = polynomial_reg.fit_transform(x)

linear_reg = LinearRegression()
linear_reg.fit(xPower2,y)

y_head = linear_reg.predict(np.array(xPower2))

plt.plot(x,y_head,color="red",label="polynomial")

plt.scatter(x, y)
plt.xlabel("Valıe")
plt.ylabel("Speed")
plt.show()

