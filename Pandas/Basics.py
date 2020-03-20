import pandas as pd

dictionary = {"NAME": ["Ali", "Veli", "Kenan", "Ayse", "Evren"],
              "AGE": [15, 18, 25, 53, 32],
              "MAAS": [1980, 2341, 2553, 800, 15400]}
dataFrame1 = pd.DataFrame(dictionary)

head = dataFrame1.head()
tail = dataFrame1.tail()

print(dataFrame1.columns)
print(dataFrame1.info())
print(dataFrame1.dtypes)
print(dataFrame1.describe())


print(dataFrame1["NAME"])

print(dataFrame1.AGE)
dataFrame1["NEW_Feature"] = [-25, -8, -1998, 3, -5]
print(dataFrame1.loc[:, "NEW_Feature"])
print(dataFrame1.loc[: 3, "NEW_Feature"])
print(dataFrame1.loc[: 3, "NAME": "NEW_Feature"])
print(dataFrame1.loc[: 3, ["NAME", "NEW_Feature"]])
print(dataFrame1.loc[::-1, ::-1])
print(dataFrame1.loc[:, :"NAME"])
print(dataFrame1.iloc[:, 0])
