import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
* maas = b0 + b1*deneyim + b2*yas

"""

df = pd.read_csv("multiple_LR.csv", sep=";")

x = df.iloc[:, [0, 2]].values
y = df.maas.values.reshape(-1, 1)

multiple_LR = LinearRegression()
multiple_LR.fit(x, y)

print("b0 is: ",multiple_LR.intercept_)
print("b1,b2 are: ",multiple_LR.coef_)

multiple_LR.predict(np.array([[10,35],[5,35]]))
print(multiple_LR.predict(np.array([[10,35],[5,35]])))