import numpy as np
import pandas as pd
import basic_func as func
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

(listing_id_train, features_train, values_train, data_train1) = func.load_unicef_data('trainxxx.json')
(listing_id_train2, features_train2, values_train2, data_train2) = func.load_unicef_data('test.json')
(listing_id_test, features_test, values_test, data_test1) = func.load_unicef_data('new_test.json')
# print all features

print(features_train)
# get train data

p = features_train.index('interest_level')
target = values_train[:, p]
# values_train = np.delete(values_train, p, 1)
# features_train.remove('interest_level')
# # get target data
# p = features_train.index('distance')
# values_train = np.delete(values_train, p, 1)
#
# p = features_test.index('distance')
# values_test = np.delete(values_test, p, 1)

train = values_train
# improvement1 -- scale the train and test data first
sc = StandardScaler()
sc.fit(train)
train = sc.transform(train)
test = sc.transform(values_test)
# test = values_test

# initialize model
# lr_model = LogisticRegression(penalty="none")
lr_model = LogisticRegression(C=500, penalty="l2", max_iter=300, tol=0.1)
# use cross-validation to take a look at accuracy
scores = cross_val_score(lr_model, train, target, cv=10)

# 10-fold cross-validation 
kf = KFold(n_splits=10)
index = []
scores_train = []
scores_valid = []
models = []
for train_index, validate_index in kf.split(train):
    X_train, X_valid = train[train_index], train[validate_index]
    y_train, y_valid = target[train_index], target[validate_index]
    temp_model = lr_model.fit(X_train, y_train)
    models.append(temp_model)
    # score for validation 
    score = temp_model.score(X_valid, y_valid)
    scores_valid.append(score)
    # score for train to observe whether overfitting occurs
    score = temp_model.score(X_train, y_train)
    scores_train.append(score)
    index.append([train_index, validate_index])
print("train scores:"), print(scores_train)
print("validation scores:"), print(scores_valid)
# pick the last model with highest validation accuracy
max_index = scores_valid.index(max(scores_valid))
last_model = models[max_index]

# fit the model with train data
# last_model.fit(train, target)

# predict the interest level in test dataset
pred1 = last_model.predict_proba(test)

# generate result
# get listing_id
p = features_train2.index('listing_id')
list_id = values_train2[:, p].reshape((values_train2.shape[0],1))
result = np.append(list_id, pred1, axis=1)
print(result)
# to meet the requirement in kaggle
data = pd.DataFrame(result, columns=['listing_id', 'low', 'medium', 'high'])
cols = list(data)
cols.insert(1, cols.pop(cols.index('high')))
data = data.loc[:, cols]
cols.insert(2, cols.pop(cols.index('medium')))
data = data.loc[:, cols]
data.to_csv('submission.csv', index=None)




