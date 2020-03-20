import pandas as pd
import matplotlib as plt

df = pd.read_csv("original.csv")
print(df.columns)
print(df.info())
print(df.describe())

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]
print(setosa.describe())
print(versicolor.describe())

# %% Matplotlib

import matplotlib.pyplot as plt
df1 = df.drop(["Id"], axis=1)
# df1.plot()
# plt.show()

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.plot(setosa.Id,
         setosa.PetalLengthCm,
         color="red",
         label="Setosa - PetalLength",
         linestyle=":")
plt.plot(versicolor.Id,
         versicolor.PetalLengthCm,
         color="green",
         label="Versicolor - PetalLength",
         linestyle=":",
         alpha=0.5)
plt.plot(virginica.Id,
         virginica.PetalLengthCm,
         color="blue",
         label="Virginica - PetalLength",
         linestyle=":")
plt.xlabel("Id")
plt.ylabel("PetalLength")
plt.legend()
plt.show()

# %% Scatter plot

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.scatter(setosa.PetalLengthCm,
            setosa.PetalWidthCm,
            color="red",
            label="Setosa")
plt.scatter(versicolor.PetalLengthCm,
            versicolor.PetalWidthCm,
            color="green",
            label="Versicolor")
plt.scatter(virginica.PetalLengthCm,
            virginica.PetalWidthCm,
            color="blue",
            label="Virginica")
plt.legend()
plt.xlabel("Length")
plt.ylabel("Width")
plt.title("Setosa Plot")
plt.show()

# %% Histogram plot
plt.hist(setosa.PetalLengthCm, bins=15)
plt.legend()
plt.xlabel("PedalCm Values")
plt.ylabel("Frequences")
plt.show()

# %% Bar Plot

import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 16, 7, 8, 9, 11])
y = x * 2 + 5

plt.bar(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("x-y label")
plt.show()
