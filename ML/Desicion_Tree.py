""" 
* CART: Classification and Regression Tree
* Information Entropy
* Terminal leaf node
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("desicion_tree.csv", sep=";", header=None)

x_axis = df.iloc[:, 0].values.reshape(-1, 1)
y_axis = df.iloc[:, 1].values.reshape(-1, 1)


tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_axis, y_axis)


x_trues = np.arange(min(x_axis), max(x_axis), 0.01).reshape(-1, 1)
y_head = tree_reg.predict(x_trues)

plt.scatter(x_axis, y_axis, color="red")
plt.plot(x_trues, y_head, color="blue")
plt.show()
