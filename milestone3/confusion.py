import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    StackingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import basic_func as func
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

(listing_id_train, features_train, values_train, data_train1) = func.load_unicef_data("new_train.json")
(listing_id_train2, features_train2, values_train2, data_train2) = func.load_unicef_data('test.json')
(listing_id_test, features_test, values_test, data_test1) = func.load_unicef_data("new_test.json")

# get target --> target
p = features_train.index('interest_level')
target = values_train[:, p]
# print(target)

# get train data --> values_train
values_train_temp1 = values_train[:, :p]
values_train_temp2 = values_train[:, p+1:]
values_train = np.append(values_train_temp1, values_train_temp2, axis=1)
train = values_train
# print(train)

# scale
print("-----scaling----")
sc = StandardScaler()
sc.fit(train)
train = sc.transform(train)
test = sc.transform(values_test)
print("----finish scale----")

X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.25)
# fit model
# lr_model = LogisticRegression(C=500, penalty="l2", max_iter=300, tol=0.1)
# lr_model = lr_model.fit(X_train, y_train)
# clf = tree.DecisionTreeClassifier(min_samples_split=0.03)
# clf = clf.fit(train, target)

gdbt_clf =  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0,
                                       min_samples_split=0.02)
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=0.03)),
    ('gdbt', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0,
                                       min_samples_split=0.02)),
    #('xgb', xgb_clf),
    ('ada', AdaBoostClassifier(n_estimators=100)),
    ('network',
     MLPClassifier(solver='lbfgs', random_state=1, activation='tanh', alpha=1e-6, hidden_layer_sizes=(10, 30, 5))),
    # ('knn', KNeighborsClassifier(n_neighbors=10)),
    ('log', LogisticRegression(C=500, penalty="l2", max_iter=300, tol=0.1)),
    ('bagging', BaggingClassifier(DecisionTreeClassifier(min_samples_split=0.03), max_samples=0.8, max_features=0.8))]
NN_model = StackingClassifier(
     estimators=estimators, final_estimator=gdbt_clf, cv=5)
# NN_model = MLPClassifier(solver='lbfgs', random_state=1, activation='tanh', alpha=1e-6, hidden_layer_sizes=(10,30,5), max_iter=500)
NN_model = NN_model.fit(train, target)
pred1 = NN_model.predict(X_valid)


C2= confusion_matrix(y_valid, pred1, labels=[0, 1, 2])
print(C2)

# plot
sns.set()
f, ax = plt.subplots()
sns.heatmap(C2, annot=True, ax=ax)
ax.set_title('confusion matrix for stacking')
ax.set_xlabel('predict value')
ax.set_ylabel('true value')
plt.show()