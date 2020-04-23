import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

prod_per_year = df.groupby("year").totalprod.mean().reset_index()
X = prod_per_year["year"]
X = X.values.reshape(-1,1)

y = prod_per_year["totalprod"]
plt.scatter(X,y)


regr = LinearRegression()
regr.fit(X,y)
print(regr.coef_,regr.intercept_)
y_predict = regr.predict(X)
plt.plot(X,y_predict,color="r")
X_future = np.array(range(2013,2051))
X_future = X_future.reshape(-1,1)

future_predict = regr.predict(X_future)
plt.plot(X_future,future_predict,color="g")

plt.show()

