"""
    Reference:
        - https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

sns.set_palette('husl')

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../Midterm/dataset/data/iris.csv')

print(data.head(), '\n')
print(data.info(), '\n')
print(data.describe(), '\n')
print(data['variety'].value_counts(), '\n')
tmp = data.drop('id', axis=1)
g = sns.pairplot(tmp, hue='variety', markers='+')
plt.show()

g = sns.violinplot(y='variety', x='sepal.length', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='sepal.width', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='petal.length', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='petal.width', data=data, inner='quartile')
plt.show()

X = data.drop(['id', 'variety'], axis=1)
y = data['variety']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)

# experimenting with different n values
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# experimenting with different n values
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, logreg.predict(X_test)))
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
print(knn.predict([[6, 3, 4, 2]]))