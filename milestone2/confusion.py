import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import basic_func as func
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.preprocessing import StandardScaler

(listing_id_train, features_train, values_train, data_train1) = func.load_unicef_data('trainxxx.json')
(listing_id_train2, features_train2, values_train2, data_train2) = func.load_unicef_data('test.json')
(listing_id_test, features_test, values_test, data_test1) = func.load_unicef_data('new_test.json')
# print all features
# print(features_train)
# get train data
p = features_train.index('interest_level')
train1 = values_train[:, :p]
train2 = values_train[:, p+1:]
train = np.append(train1, train2, axis=1)
# get target data
target = values_train[:, p]

# scale
sc = StandardScaler()
sc.fit(train)
train = sc.transform(train)
test = sc.transform(values_test)

X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.25)
# fit model
# lr_model = LogisticRegression(C=500, penalty="l2", max_iter=300, tol=0.1)
# lr_model = lr_model.fit(X_train, y_train)
clf = tree.DecisionTreeClassifier(min_samples_split=0.03)
clf = clf.fit(train, target)

pred1 = clf.predict(X_valid)
C2= confusion_matrix(y_valid, pred1, labels=[0, 1, 2])
print(C2)

# plot
sns.set()
f, ax = plt.subplots()
sns.heatmap(C2, annot=True, ax=ax)
ax.set_title('confusion matrix for decision tree')
ax.set_xlabel('predict value')
ax.set_ylabel('true value')
