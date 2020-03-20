import numpy as np


def main():
    # Element - wise, Array - wise

    first = np.random.randint(1, 10, size=5)
    print(first)

    second = np.random.randint(1, 10, size=5)
    print(second)

    print(first + second)
    print(np.add(first, second))

    print(type(first + second))
    print(type(np.add(first, second)))

    print(first - second)
    print(np.subtract(first, second))

    print(first * second)
    print(np.multiply(first, second))

    print(first / second)
    print(np.divide(first, second))

    print(first + 3)
    print(first - 3)
    print(first * 3)
    print(first / 3)

    print(first == second)
    print(np.array_equal(first, second))

    print(np.sqrt(first))
    print(np.power(first, 3))
    print(np.exp(first))

    fn = np.random.randint(1, 10, size=9).reshape(3, 3)

    print(fn)
    print(np.sum(fn))
    print(fn.sum(axis=1))
    print(fn.sum(axis=0))
    print(fn.min())
    print(fn.max())
    print(fn.argmax())
    print(fn.argmin())


if __name__ == "__main__":
    main()