import pandas as pd
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import graphviz
import basic_func as func
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

(listing_id, features, values, data) = func.load_unicef_data("trainxxx.json")
(listing_id2, features2, values2, data2) = func.load_unicef_data("test.json")
(listing_id_test1, features_test1, values_test1, data_test1) = func.load_unicef_data("new_test.json")
# (listing_id_test2, features_test2, values_test2, data_test2) = func.load_unicef_data("new_test2.json")
p = features.index('interest_level')
X = values[:, :p]
X = np.append(X, values[:, p+1:], axis=1)
y = values[:, p]
t = list()
row = y.shape[0]
print(row)
count1 = 0
count2 = 0
for i in range(row):
    if y[i] == 0:
        t.append('low')
    elif y[i] == 2:
        t.append('high')
    elif y[i] == 1:
        t.append('medium')
t = np.array(t)
# X, y = load_iris(return_X_y=True)
# X_train, X_valid, y_train, y_valid = train_test_split(X, t, test_size=0.25)
# min_samples_split=0.03
clf = tree.DecisionTreeClassifier(min_samples_split=0.03)
print(cross_val_score(clf, X, t, cv=10, scoring='roc_auc_ovr'))


kf = KFold(n_splits=10)
index = []
scores = []
train_scores2 = []
clfs = []
for train_index, validate_index in kf.split(X):
    X_train, X_validate = X[train_index], X[validate_index]
    y_train, y_validate = t[train_index], t[validate_index]
    clf = clf.fit(X_train, y_train)
    clfs.append(clf)
    score = clf.score(X_validate, y_validate)
    scores.append(score)
    print(score)
    score = clf.score(X_train, y_train)
    train_scores2.append(score)
    index.append([train_index, validate_index])
    print(score)
max_index = scores.index(max(scores))


# use the best validation to prune the tree
temp = index[max_index]
X_train, X_validate = X[temp[0]], X[temp[1]]
y_train, y_validate = t[temp[0]], t[temp[1]]
clf = tree.DecisionTreeClassifier(min_samples_split=0.03)
path = clf.cost_complexity_pruning_path(X, t)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")
# plt.show()

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(min_samples_split=0.03, ccp_alpha=ccp_alpha)
    clf.fit(X, t)
    clfs.append(clf)
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X, t) for clf in clfs]
max_index = test_scores.index(max(test_scores))
clf = clfs[max_index]

# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train",
#         drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test",
#         drawstyle="steps-post")
# ax.legend()
# plt.show()

clf = clfs[max_index]
# print(clf.score(X_validate, y_validate))

# scores = cross_val_score(clf, X, t, cv=5)

# clf.score(X_train, y_train)
# clf.score(X_valid, y_valid)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree")
# p = features_test1.index('access_des')
result = clf.predict_proba(values_test1)
# result2 = clf.predict_proba(values_test2)
#
# result = np.append(result1, result2, axis=0)
p = features2.index('listing_id')
list_id = values2[:, p].reshape((values2.shape[0], 1))
result = np.append(list_id, result, axis=1)
print(result)
data = pd.DataFrame(result, columns=['listing_id', 'high', 'low', 'medium'])
cols = list(data)
cols.insert(2, cols.pop(cols.index('medium')))
data = data.loc[:, cols]
data.to_csv('submission.csv', index=None)
