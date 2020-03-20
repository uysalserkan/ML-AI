import numpy as np

data1 = np.array([2, 5], dtype='uint32')
data2 = np.array([2, 5], dtype=float)
data3 = np.array([2, 5], dtype=complex)
data4 = np.array([2, 5], dtype=bool)
data5 = np.array([2, 5], dtype=np.int)

print(
    f"Data1 Type: {data1.dtype}\nData2 Type: {data2.dtype}\nData3 Type: {data3.dtype}\nData4 Type: {data4.dtype}\nData5 Type: {data5.dtype}\n"
)

y = np.array([2, 5], dtype=int)
print(y.dtype)

y = np.array(y, dtype=float)
print(y.dtype)

y = y.astype(dtype=complex)
print(y.dtype)

z = np.sqrt(np.array([-1, 9, 25], dtype=float))
print(f"Data type: {z.dtype}\nDatas: {z}")

np.save("theZarray", z)

f = np.load("theZarray.npy")
print(f)
