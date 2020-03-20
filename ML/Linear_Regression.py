import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
"""
 * Line fit || Linear regression
 ! y = b0 + b1 * x
 ? b0 = Constant or bias
 b1 = coefficient or incline


 ! residual = y - y_head (or y^)

 MSE = Mean Squared Error
 n = Number of sample
 MSE = sum(residual^2)/n
 ! Our purpose is that find the minimum MSE for best predictions
"""


def main():
    df = pd.read_csv("original.csv", sep=";")
    # plt.scatter(df.deneyim, df.maas)
    # plt.xlabel("Deneyim")
    # plt.ylabel("Maas")
    # plt.show()

#  Linear Regression
    linear_reg = LinearRegression()
    first = df.deneyim.values.reshape(-1, 1)
    second = df.maas.values.reshape(-1, 1)
    print("#####\n\nshapes", first.shape, second.shape)
    linear_reg.fit(first, second)

# Prediction
    print("b0 is: ", linear_reg.predict([[0]]))
    print("b0 is: ", linear_reg.intercept_)
    print("b1 is: ", linear_reg.coef_)

    test_years = np.array(
        [0,1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13,14,15]).reshape(-1, 1)
    plt.scatter(first, second)
    # plt.show()
    y_head = linear_reg.predict(test_years)

    plt.plot(test_years, y_head, color="red")
    plt.show()

    # ! R-Square
    print("r2 score: ", r2_score(second, y_head))


if __name__ == "__main__":
    main()
