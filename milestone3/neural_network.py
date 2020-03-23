import basic_func as func
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time as time

print("----read data-----")
(listing_id_train, features_train, values_train, data_train1) = func.load_unicef_data("new_train.json")
(listing_id_train2, features_train2, values_train2, data_train2) = func.load_unicef_data('test.json')
(listing_id_test, features_test, values_test, data_test1) = func.load_unicef_data("new_test.json")
print("-----finish read data-----")

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

# improvement1 -- scale the train and test data first
print("-----scaling----")
sc = StandardScaler()
sc.fit(train)
train = sc.transform(train)
test = sc.transform(values_test)
print("----finish scale----")

# grid_searchcv -- for choosing best parameter
print("----grid search cv----")
clf_nn = MLPClassifier(solver='lbfgs', random_state=1)
params = {
    'alpha': [1e-6, 1e-5, 1e-4],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
    'hidden_layer_sizes': [(10, 30, 5),(30, 30, 5), (20, 20, 20), (30, 30, 5)]
}
gs_nn = GridSearchCV(clf_nn, param_grid=params, scoring='neg_log_loss', n_jobs=2, cv=2, verbose=2, refit=True) # cv=5
start = time.time()
gs_nn.fit(values_train, target)
print('- Time: %.2f minutes' % ((time.time() - start)/60))
print('- Best score: %.4f' % gs_nn.best_score_)
print('- Best params: %s' % gs_nn.best_params_)
print("---grid search cv finish---")
# perform 10 cross-validation
best_model = MLPClassifier(solver='lbfgs', random_state=1, activation='tanh', alpha=1e-6, hidden_layer_sizes=(10,30,5), max_iter=500)
kf = KFold(n_splits=10)
index = []
scores_train = []
scores_valid = []
models = []
for train_index, validate_index in kf.split(train):
    X_train, X_valid = train[train_index], train[validate_index]
    y_train, y_valid = target[train_index], target[validate_index]
    temp_model = best_model.fit(X_train, y_train)
    models.append(temp_model)
    # score for validation data
    score = temp_model.score(X_valid, y_valid)
    scores_valid.append(score)
    # scores for training data
    score = temp_model.score(X_train, y_train)
    scores_train.append(score)
    index.append([train_index, validate_index])
print("train scores:"), print(scores_train)
print("validation scores:"), print(scores_valid)
# pick the last model with highest validation accuracy
max_index = scores_valid.index(max(scores_valid))
last_model = models[max_index]

# predict probability
# last_model = MLPClassifier(solver='lbfgs', random_state=1, activation='tanh', alpha=1e-6, hidden_layer_sizes=(10,30,5), max_iter=500)
# last_model = last_model.fit(train, target)
pred1 = last_model.predict_proba(test)

# get listing_id
p = features_train2.index('listing_id')
list_id = values_train2[:, p].reshape((values_train2.shape[0],1))
result = np.append(list_id, pred1, axis=1)
print(result)

# change order to meet the requirement in kaggle
data = pd.DataFrame(result, columns=['listing_id', 'low', 'medium', 'high'])
cols = list(data)
cols.insert(1, cols.pop(cols.index('high')))
data = data.loc[:, cols]
cols.insert(2, cols.pop(cols.index('medium')))
data = data.loc[:, cols]

# generate result
print("----write data----")
data.to_csv('submission.csv', index=None)
print("---write finish---")
