import numpy as np


def main():
    ar1 = np.random.randint(low=1, high=25, size=20)
    ar2 = np.random.randint(low=1, high=25, size=20)

    ar3 = np.intersect1d(ar1, ar2)
    print(ar3)
    print(type(ar3))
    print(np.setdiff1d(ar1, ar2))
    print(np.setdiff1d(ar2, ar1))
    print(np.union1d(ar1, ar2))
    print(np.in1d(ar1, ar2))
    print(np.unique(ar1))
    print(np.unique(ar2))

    print("Sort")
    print(np.sort(ar1))  # Function
    print(ar1)

    ar1.sort()  # Method
    print(ar1)

    xfs = np.random.randint(1, 10, size=(5, 5))
    print(xfs)
    print(np.sort(xfs, axis=0))  # Column
    print(np.sort(xfs, axis=1))  # Row


if __name__ == "__main__":
    main()