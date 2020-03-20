import numpy as np


def main():
    x = np.linspace(1, 21, 11)
    print(x)

    y = np.array([2, 4, 7])
    print(y)

    print(x[y])

    z = np.arange(25).reshape(5, 5)
    print(z)
    k = np.array([2, 4])
    print(k)

    print(z[k, :])
    print(z[:, k])

    m = z[:, k]
    m[0, 0] = 100
    print(m)
    print(z)
    # ! 'Fancy Indexing
    # Bir ndArray'i başka bir array ile kopyalarsak oluşan değişiklik local kalır
    print(np.shares_memory(z, m))

    # ! 'Boolean Indexing
    f = np.arange(10)
    print(f)
    print(f[(f % 2 == 0)])
    p = f[(f % 2 == 0)]
    p[0] = 100
    print(p)
    print(f)

    s1 = np.random.randint(10, size=10)
    s2 = np.random.randint(10, size=10)
    print(s1)
    print(s2)
    print(s1 > s2)
    print(s1 < s2)
    print(type(s1 < s2))
    print((s1 > s2).dtype)

    print(np.all(s1 > s2))
    print(np.any(s1 > s2))

    t = np.linspace(1, 21, 11)
    print(t)

    mask = (t % 3 == 0)
    print(mask)
    print(type(mask))
    print(t[mask])


if __name__ == "__main__":
    main()