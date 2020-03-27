from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Class 1
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)
# Class 2
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)
# Class 3
x3 = np.random.normal(45, 5, 1000)
y3 = np.random.normal(35, 5, 1000)

x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

dictionary = {"x": x, "y": y}

df = pd.DataFrame(dictionary)

""" 
# plt.scatter(x1, y1, color="green")
# plt.scatter(x2, y2, color="green")
# plt.scatter(x3, y3, color="green")
# plt.show()

wcss = []

for each in range(1, 15):
    kmenas = KMeans(n_clusters=each)
    kmenas.fit(df)
    wcss.append(kmenas.inertia_)
# plt.plot(range(1, 15), wcss)
# plt.xlabel("number of test")
# plt.ylabel("score")
# plt.show()
# ! in the plot the k value must be 3

KM = KMeans(n_clusters=3)
clusters = KM.fit_predict(df)
df["Label"] = clusters
print(df.sample(10))

# Visualization
plt.scatter(df.x[df.Label == 0], df.y[df.Label == 0],color="red",label="Class 1")
plt.scatter(df.x[df.Label == 2], df.y[df.Label == 2],color="green",label="Class 2")
plt.scatter(df.x[df.Label == 1], df.y[df.Label == 1],color="blue",label="Class 3")
plt.scatter(KM.cluster_centers_[:,0],KM.cluster_centers_[:,1],color="black",label="Centers")
plt.legend()
plt.show() """

# Dendogram

""" merg = linkage(df, method="ward")
dendrogram(merg, leaf_rotation=90)
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()
 """

 # Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
cluster = hc.fit_predict(df)
df["LABEL"] = cluster

plt.scatter(df.x[df.LABEL == 0], df.y[df.LABEL == 0],color="red",label="Class 1")
plt.scatter(df.x[df.LABEL == 2], df.y[df.LABEL == 2],color="green",label="Class 2")
plt.scatter(df.x[df.LABEL == 1], df.y[df.LABEL == 1],color="blue",label="Class 3")
plt.legend()
plt.show()