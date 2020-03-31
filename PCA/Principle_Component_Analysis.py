from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
data = iris.data
y = iris.target
feature_names = iris.feature_names

df = pd.DataFrame(data, columns=feature_names)
df["sinif"] = y

x = data

pca = PCA(n_components=2, whiten=True)  # Whiten << normalization
pca.fit(x)

x_pca = pca.transform(x)

print("variance ratio: ", pca.explained_variance_ratio_)
print("sum: ", sum(pca.explained_variance_ratio_))


# ? 2D Visualition
df["p1"] = x_pca[:, 0]
df["p2"] = x_pca[:, 1]

color = ["red", "green", "purple"]
for each in range(3):
    plt.scatter(df.p1[df.sinif == each], df.p2[df.sinif ==
                                               each], label=iris.target_names[each])

plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
