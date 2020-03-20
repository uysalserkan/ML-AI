import pandas as pd
import numpy as np
# %% List Comprehension
dictionary = {"NAME": ["Ali", "Veli", "Kenan", "Ayse", "Evren"],
              "AGE": [15, 18, 25, 53, 32],
              "MAAS": [1980, 2341, 2553, 800, 15400]}
dataFrame1 = pd.DataFrame(dictionary)

mean_value = (dataFrame1.MAAS.mean())
print(mean_value)
print(np.mean(dataFrame1.MAAS))
dataFrame1["maas_seviyesi"] = ["dusuk" if mean_value >
                               each else "yuksek" for each in dataFrame1.MAAS]
dataFrame1.columns = [each.lower() for each in dataFrame1.columns]
dataFrame1.columns = [each.split[0]+"_"+each.split[1]
                      if len(each.split()) > 1 else each for each in dataFrame1.columns]

dataFrame1.drop(["AGE"], axis=1, inplace=True)

data1 = dataFrame1.head()
data2 = dataFrame1.tail()

# Vertical

data_concat_vertical = pd.concat([data1, data2], axis=0)
data_concat_horizontal = pd.concat([data1, data2], axis=1)
