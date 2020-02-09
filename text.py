import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func



(listing_id, features, values, train_df) = func.load_unicef_data(fname="train.json")

# fina all the description data
words_all = func.deal_with_text(features, values, "description", 0.1)

# transform the description part
p = features.index("description")
descriptions = list()
for i in range(values[:, p].shape[0]):
    description = list()
    # find the words in every description
    if values[i, p].replace(" ", "") != "" and values[i, p] != "Must see!" and values[i, p].replace(".", "") != "":
        words = func.deal_with_text(features, values[i, :].reshape((1, values.shape[1])), "description")
        for word in words:
            if word in words_all:
                description.append(word)
    # append the new description
    descriptions.append(description)










