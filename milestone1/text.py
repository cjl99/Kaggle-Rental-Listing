import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from tkinter import _flatten

(listing_id, features_all, values, train_df) = func.load_unicef_data(fname="final_train.json")

# fina all the description data
words_all = func.deal_with_text(features_all, values, "description", 0.1)
# transform the description part
p = features_all.index("description")
descriptions = list()
for i in range(values[:, p].shape[0]):
    description = list()
    # find the words in every description
    if values[i, p].replace(" ", "") != "" and values[i, p] != "Must see!" and values[i, p].replace(".", "") != "":
        words = func.deal_with_text(features_all, values[i, :].reshape((1, values.shape[1])), "description")
        for word in words:
            if word in words_all:
                description.append(word)
    # append the new description
    descriptions.append(description)

values[:, p] = descriptions
print("description = "+str(len(descriptions)))


ps = PorterStemmer()
p = features_all.index('features')
features = values[:, p]
features = list(features)

# find all features
new_features = list(_flatten(features))
for i in range(len(new_features)):
    new_features[i] = new_features[i].lower()
    new_features[i] = ps.stem(new_features[i])

# find the high frequency features
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
values[:, p] = new_values
print("features = "+str(len(new_values)))

# generate the new data file
data = func.generate_new_json(listing_id, features_all, values, "new_data.json")
print(data)

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
