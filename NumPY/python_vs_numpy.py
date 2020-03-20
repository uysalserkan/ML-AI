import numpy as np
import time

num_start_time = time.time()
arr1 = np.arange(10000000)
arr1**2
num_total_time = time.time() - num_start_time

python_start_time = time.time()
arr2 = range(10000000)
[element**2 for element in arr2]
python_total_time = time.time() - python_start_time

print(
    f"Numpy Time is: {num_total_time}\nPython Time is: {python_total_time}\n")
