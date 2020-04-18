from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

x = (x-np.min(x))/(np.max(x)-np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

knnModel = KNeighborsClassifier(n_neighbors=3)

# 10 fold cross validation
acc = cross_val_score(estimator=knnModel, X=x_train, y=y_train, cv=10)

print("Average accuracy: ", np.mean(acc))
print("Average standart derivation: ", np.std(acc))

knnModel.fit(x_train, y_train)

print("Test Accuracy: ", knnModel.score(x_test, y_test))

# Grid Search cross-validation

grid = {"n_neighbors": np.arange(1, 75)}

knnM = KNeighborsClassifier()

knnM_cv = GridSearchCV(knnM, grid, cv=10)

knnM_cv.fit(x, y)

print("tuned hyperparameters K: ", knnM_cv.best_params_)
print("tuned hyperparameters'e göre en iyi skor: ", knnM_cv.best_score_)


# %% Grid search cross validation with logistic regression

x = x[:100, :]
y = y[:100]


# L1 = lasso , L2 = Ridge
param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
logreg_cv.fit(x, y)

print("tuned hyperparameters: ", logreg_cv.best_params_)
print("accuracy: ", logreg_cv.best_score_)
