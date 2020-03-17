print('This is a decision tree test file')
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

# Import data set
df = pd.read_csv('G:/python/Machine Learning/data/hired_data.csv', header=0)
print(df.columns)

# Process the dataset
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Interned'] = df['Interned'].map(d)
df['Top-tier School'] = df['Top-tier School'].map(d)
df['Employed'] = df['Employed'].map(d)

edu = {'BSc': 1, 'MSc': 2, 'PHD': 3}
df['Level of Education'] = df['Level of Education'].map(edu)

print(df.head())

features = list(df.columns[1:7])
x = df[features]
y = df['Hired']

classifyer = DecisionTreeClassifier()
model = classifyer.fit(x, y)

#
# from sklearn.externals.six import StringIO
# from IPython.display import Image, display
# from sklearn.tree import export_graphviz
# import pydot
#
# dot_data = StringIO()
# export_graphviz(model, out_file=dot_data, feature_name=features)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())


# Random Forest
print(x)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=10)
rf_clf = rf_clf.fit(x, y)

print(rf_clf.predict([10, 1, 4, 0, 0, 0]))
