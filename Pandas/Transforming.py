import pandas as pd
import numpy as np
# %% Transforming Data
dictionary = {"NAME": ["Ali", "Veli", "Kenan", "Ayse", "Evren"],
              "AGE": [15, 18, 25, 53, 32],
              "MAAS": [1980, 2341, 2553, 800, 15400]}
dataFrame1 = pd.DataFrame(dictionary)

dataFrame1["list_comp"] = [each * 2 for each in dataFrame1.AGE]


def multiply(age):
    return age*2


dataFrame1["apply_method"] = dataFrame1.AGE.apply(multiply)
