import numpy as np


def main():
    x = np.random.randint(1, 10, size=(8, ))
    print(x)
    print(x.mean())
    print(np.median(x))
    print(x.std())

    y = np.random.randint(1, 10, size=(4, 3))
    print(y)
    print(y.mean(axis=0))
    print(y.mean(axis=1))
    print(np.median(y, axis=0))
    print(np.median(y, axis=1))
    print(y.std(axis=0))
    print(y.std(axis=1))

    first = np.random.randint(1, 10, size=(1, 3))
    second = np.random.randint(1, 10, size=(3, 1))
    third = np.random.randint(1, 10, size=(3))

    print(first)
    print(second)
    print(first + second)
    print(third + second)


if __name__ == "__main__":
    main()