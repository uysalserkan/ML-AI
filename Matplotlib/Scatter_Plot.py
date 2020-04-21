import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

mu = 1
std = 0.5
mu2 = 4.188

np.random.seed(100)

xs = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.75,std,100)), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100))

ys = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100)), np.random.normal(0.75,std,100))

values = list(zip(xs, ys))

model = KMeans(init='random', n_clusters=2)

results = model.fit_predict(values)

plt.scatter(xs, ys, c=results, alpha=0.6)

colors = ['#6400e4', '#ffc740']

for i in range(2):
  points = np.array([values[j] for j in range(len(values)) if results[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.6)
  
plt.title('Codecademy Mobile Feedback - Data Science')

plt.xlabel('Learn Python')
plt.ylabel('Learn SQL')
  
plt.show()
