import pandas as pd
import  numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston
boston = load_boston()
# print(boston)

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)

df_y = pd.DataFrame(boston.target)

# print(df_x.describe())

reg_model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.2, random_state=4)
reg_model.fit(x_train, y_train)
reg_model.coef_ # To check each features weight

predict = reg_model.predict(x_test)
print(predict)

# Mean Square error
mean_value = np.mean(predict - y_test)
print(mean_value)