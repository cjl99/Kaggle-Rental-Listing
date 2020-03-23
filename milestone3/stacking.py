import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import basic_func as func
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

(listing_id, features, values, data) = func.load_unicef_data("new_train.json")
(listing_id2, features2, values2, data2) = func.load_unicef_data("test.json")
(listing_id_test1, features_test1, values_test1, data_test1) = func.load_unicef_data("new_test.json")

p = features.index('interest_level')
y = values[:, p]
values = np.delete(values, p, 1)
features.remove('interest_level')
p = features.index('distance')
X = np.delete(values, p, 1)

p = features_test1.index('distance')
values_test1 = np.delete(values_test1, p, 1)

# t = list()
# row = y.shape[0]
# print(row)
#
# for i in range(row):
#     if y[i] == 0:
#         t.append('low')
#     elif y[i] == 2:
#         t.append('high')
#     elif y[i] == 1:
#         t.append('medium')
# t = np.array(t)
t = y

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
values_test1 = sc.transform(values_test1)

# q = log(47400)+1
# clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=0.03)
# clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=0.03)
gdbt_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0,
                                      min_samples_split=0.02)
# clf = KNeighborsClassifier(n_neighbors=5)
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', colsample_bytree=1, learning_rate=0.02,
                            max_depth=6, alpha=10, n_estimators=100, min_samples_split=0.02, silent=1, subsample=0.7,
                            eval_metric="mlogloss")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=0.03)),
    ('gdbt', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0,
                                       min_samples_split=0.02)),
    ('xgb', xgb_clf),
    ('ada', AdaBoostClassifier(n_estimators=100)),
    ('network',
     MLPClassifier(solver='lbfgs', random_state=1, activation='tanh', alpha=1e-6, hidden_layer_sizes=(10, 30, 5))),
    # ('knn', KNeighborsClassifier(n_neighbors=10)),
    ('log', LogisticRegression(C=500, penalty="l2", max_iter=300, tol=0.1)),
    ('bagging', BaggingClassifier(DecisionTreeClassifier(min_samples_split=0.03), max_samples=0.8, max_features=0.8))]
clf = StackingClassifier(
     estimators=estimators, final_estimator=gdbt_clf, cv=5)

# clf = clf.fit(X, t)
# score = clf.score(X, t)
# print(score)
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
clf = clfs[max_index]

result = clf.predict_proba(values_test1)
p = features2.index('listing_id')

list_id = values2[:, p].reshape((values2.shape[0], 1))
result = np.append(list_id, result, axis=1)
print(result)
data = pd.DataFrame(result, columns=['listing_id', 'low', 'medium', 'high'])
# data = pd.DataFrame(result, columns=['listing_id', 'high', 'low', 'medium'])
# cols = list(data)
# cols.insert(2, cols.pop(cols.index('medium')))
# data = data.loc[:, cols]
data.to_csv('submission.csv', index=None)
