import numpy as np


def main():
    x = np.arange(10)
    print(x)
    print(x[2:5])
    print(x[2:])
    print(x[:2])
    print(x[:-3])
    print(x[-3:])

    y = np.arange(25).reshape(5, 5)
    print(y)
    print(y[0:3, :-3])
    print(y[::])

    z = np.arange(9).reshape(3, 3)
    k = z[0:2, :2]
    print(k[::])
    k[0, 0] = 2508
    print(k)
    print(z)
    print(np.shares_memory(z, k))

    m = np.arange(16).reshape(4, 4)
    print(m)
    n = np.copy(m[0:3, 1:3])
    print(n)
    print(np.shares_memory(n, m))
    n2 = m[1:2, 0:1].copy()
    print(n2)
    print(np.shares_memory(n2, m))


if __name__ == "__main__":
    main()