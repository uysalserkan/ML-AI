import numpy as np

x = np.arange(25)

print(x[5])
print(x[-5])

x = np.reshape(x, (5, 5))
print(x[-1][-1])
y = np.delete(x, [4, 3, 2, 1, 0]).reshape(5, 4)
print(y)
y = np.delete(y, 0, 1)
print(y)
# y = np.append(y, [25, 26, 27, 28, 5], axis=1)
# print(y)
# y = np.insert(y,25,5)

arr1 = np.random.randint(3, 12, 9)
arr1 = np.reshape(arr1, (3, 3))
# arr1.dtype = 'float64'
print("The arr is", arr1)
arr2 = np.random.random(3)
print("The arr is", arr2)
arr3 = np.vstack((y, x))
print(arr3)
