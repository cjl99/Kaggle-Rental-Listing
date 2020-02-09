"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean
import math
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def load_unicef_data(fname=""):
    """Loads Unicef data from JSON file.

    Retrieves a matrix of all rows and columns from rental listing data
    dataset.

    Args:
      none

    Returns:
      listing_id,feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N listing_ids
      features: vector of F feature names
      values: matrix N-by-F
    """

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_json(fname)
    # find feature names.
    features = data.axes[1][:]
    # transform to list
    features = features.tolist()
    listing_id = data.axes[0][:]
    values = data.values[:, :]

    return (listing_id, features, values, data)

def generate_new_json(id, features, values, filename):
    data = pd.DataFrame(values, columns = features, index = id)
    data.to_json(filename)

    return data

def load_by_data(data):
    listing_id = data.axes[0][1:]
    values = data.values[:, :]

    return listing_id, values


def not_digit_and_underline(a):
    digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
    if a in digit:
        return False
    else:
        return True


def deal_with_text(features, values, feature_name, min_df=0):
    # get index first
    p = features.index(feature_name)
    corpus = values[:, p]
    # delete digit and underline
    for i in range(len(corpus)):
        corpus[i] = ''.join(filter(not_digit_and_underline, corpus[i]))
        corpus[i] = corpus[i].replace(r"<.*>", " ")
        corpus[i] = corpus[i].replace(r"<br />", " ")

    # seperate line to 1 or 2 words
    # deal with stop words
    # max_df and min_df can be determined by cross validation
    words = list()
    vectorizer = CountVectorizer(min_df=min_df, ngram_range=(1, 2), stop_words='english')
    if corpus[0] != "":
        X = vectorizer.fit_transform(corpus)
        words = vectorizer.get_feature_names()
    else:
        return words
    # stem the words
    ps = PorterStemmer()
    for i in range(len(words)):
        words[i] = ps.stem(words[i])

    # take a look
    # print(len(set(words)))
    # print(words)

    return words


def seperate_df_level(df=None):
    """seperate train_df by interest level 'low' 'medium' 'high'

    Args:
      train_df

    Returns:
      df_high, df_medium, df_low --The seperated result
    """
    high = df.loc[df["interest_level"] == "high", ["listing_id"]]
    df_high = df.loc[df["listing_id"].isin(high["listing_id"])]

    medium = df.loc[df["interest_level"] == "medium", ["listing_id"]]
    df_medium = df.loc[df["listing_id"].isin(medium["listing_id"])]

    low = df.loc[df["interest_level"] == "low", ["listing_id"]]
    df_low = df.loc[df["listing_id"].isin(low["listing_id"])]

    return df_high, df_medium, df_low
