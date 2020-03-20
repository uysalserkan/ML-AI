import numpy as np

first = np.zeros((2, 5))
second = np.ones((5, 2))
twenty5 = np.full((25, 25), 25, dtype=int)
print(
    f"First is: {first} and its dtype: {first.dtype}\nSecond is: {second} and its dtype: {second.dtype}\nOther variable: {twenty5} and its dtype: {twenty5.dtype}\n"
)
sln = 15 * np.ones((2, 5))
snl = 25 + np.zeros((2, 5))
print(sln, "\n", snl)

aslm = np.empty((2, 5))
aslm.fill(8)
print(aslm)

brm = np.eye(5)
print(brm)
brm1 = np.eye(5, k=1)
print(brm1)
brmneg2 = np.eye(5, k=-2)
print(brmneg2)

fms = np.identity(5)
print(fms)

sal = np.diag([25, 26, 24, 21])
print(sal)

fish1 = np.arange(25)
print(fish1)
fish2 = np.arange(15, 25)
print(fish2)
fish3 = np.arange(15, 25, 5)
print(fish3)

doggo1 = np.linspace(
    5, 25, 15
)  # ? Default olarak 50 veriyor kaç tane değer vericeğine, endpoint ise kapalı aralık mı açık aralık mı olduğunu belirliyor
doggo2 = np.linspace(
    5, 25, 15, endpoint=False
)  # ? Default olarak 50 veriyor kaç tane değer vericeğine, endpoint ise kapalı aralık mı açık aralık mı olduğunu belirliyor
print(doggo1)
print(doggo2)

doggo1 = np.reshape(doggo1, (3, 5))

catty1 = np.random.random((25, 5))
print(catty1)
catty2 = np.random.randint(5, 25, (25, 5))
print(catty2)
