""" 
! Ensemble Learning

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("desicion_tree.csv", sep=";", header=None)

x_axis = df.iloc[:, 0].values.reshape(-1, 1)
y_axis = df.iloc[:, 1].values.reshape(-1, 1)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_axis, y_axis)

print("7.5$: ", rf.predict([[7.8]]))

x_ = np.arange(min(x_axis),max(x_axis),0.01).reshape(-1,1)
y_head = rf.predict(x_)


plt.scatter(x_axis,y_axis,color="red")
plt.plot(x_,y_head,color="blue")
plt.show()

# ! R-Square
y_head = rf.predict(x_axis)
print("r_score: ", r2_score(y_axis, y_head))
