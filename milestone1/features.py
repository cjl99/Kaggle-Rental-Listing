import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func
import time
import seaborn as sns
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from tkinter import _flatten

(listing_id, features_all, values, train_df) = func.load_unicef_data('train.json')
# print(features)
# print(values)
ps = PorterStemmer()
p = features_all.index('features')
features = values[:, p]
features = list(features)
# new_words = list()
new_features = list(_flatten(features))
for i in range(len(new_features)):
    new_features[i] = new_features[i].lower()
    new_features[i] = ps.stem(new_features[i])

results = pd.value_counts(new_features)
temp = results.loc[results < 50]
results = results.loc[~results.isin(temp)]
feature_name = list(results.index)
# transform the features
new_values = list()
for feature in values[:, p]:
    items = list()
    for item in feature:
        item_uniform = item.lower()
        item_uniform = ps.stem(item_uniform)
        if item_uniform in feature_name:
            items.append(item_uniform)
    new_values.append(items)

# deal with display address
p = features_all.index('display_address')
features = values[:, p]
features = list(features)
# new_words = list()
new_features = list(_flatten(features))
for i in range(len(new_features)):
    new_features[i] = new_features[i].lower()
    new_features[i] = ps.stem(new_features[i])

results = pd.value_counts(new_features)
print(results)
