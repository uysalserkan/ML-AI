import pandas as pd

# %% Filtering
dictionary = {"NAME": ["Ali", "Veli", "Kenan", "Ayse", "Evren"],
              "AGE": [15, 18, 25, 53, 32],
              "MAAS": [1980, 2341, 2553, 800, 15400]}
dataFrame1 = pd.DataFrame(dictionary)

filter1 = dataFrame1.MAAS > 2000

filtered_data = dataFrame1[filter1]
print(filtered_data)
filter2 = dataFrame1.AGE < 20
filtered_data2 = dataFrame1[filter1 & filter2]
print(filtered_data2)
print(dataFrame1[dataFrame1.AGE > 30])
