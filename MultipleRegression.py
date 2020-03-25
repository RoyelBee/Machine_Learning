import pandas as pd

data = pd.read_csv('Data/cars.csv')
# print(data.head())
#
# print(data.columns)
features = data[['Volume', 'Weight']]
target = data[['CO2']]

# # Create Linear regression model
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(features, target)

# Predict model
volume = 1600
weight = 1300
predict = int(reg.predict([[volume, weight]]))
print('Your car volume = ', volume, 'and weight =', weight, 'Then your car speeds will be run around = ', predict,
      ' Km/h')
