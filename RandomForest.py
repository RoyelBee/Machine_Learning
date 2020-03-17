import pandas as pd
from sklearn.datasets import load_digits

# Load digits image dataset
digits = load_digits()
print(dir(digits))

import matplotlib.pyplot as plt
# plt.gray() # make digits images color gray as default

# Show image data 0 to 10
for i in range(11):
    plt.matshow(digits.images[i])
    plt.show()

df = pd.DataFrame(digits.data)
# print(df.head())
# print(digits.target)

df['target'] = digits.target
# print(df.head())

# Create Train and test data for both X and Y
# Here test_size=.2 is test data = 20%
# df.drop(['target'] = Drop target variable from x dataset , digits.target = y dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), digits.target, test_size=.2)

# Test all four data size length
# print('X_train= ',len(X_train))
# print('X_test = ',len(X_test))
# print('y_train = ',len(y_train))
# print('y_test = ',len(y_test))

# Create Random Forest Model
# Here n_estimators is the number of chunk sizes of RF model . Default size is 10.
# If we increase the size model accuracy will be increase
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=90)


# Fit model with x_train and y_train dataset
model.fit(X_train, y_train)

# To check the model accuracy ( y_test is tested by x_test)
score = model.score(X_test, y_test)
print('Model Score = ', score)

# Create confusion matrix to predict x_test dataset (Which are not test yet)
y_predict = model.predict(X_test)

# Create a confusion matrix to evaluate how accurately yhe model works
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# visualize the data
import seaborn as sn
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Truth represent how much number should be present in Y axis for all data (0 - 9)
# Predicted represent models accuracy results in X axis
# # If both are same then model predict 100% accurately