import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from tkinter import _flatten
import time
from geopy import distance

train_df = pd.read_json('test.json')
train_df_real = pd.read_json('train2.json')
train_df_final = pd.read_json('trainxxx.json')
# bathroom -- no change
# bedroom -- no change

# building id drop
train_df = train_df.drop('building_id', axis=1)


# created use month and hours
train_df['created'] = pd.to_datetime(train_df['created'])
train_df['created_month'] = train_df['created'].dt.month
train_df['created_day'] = train_df['created'].dt.day
train_df['created_hour'] = train_df['created'].dt.hour
train_df = train_df.drop('created', axis=1)

# decription count words
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
train_df = train_df.drop('description', axis=1)

# display address & street_address & manager_id drop
train_df['east'] = train_df['display_address'].apply(lambda x: x.find('east')>-1).astype(int)
train_df['west'] = train_df['display_address'].apply(lambda x: x.find('west')>-1).astype(int)
train_df = train_df.drop(['display_address', 'street_address'], axis=1)

# listing_id -- no change

# photos numbers
train_df['num_photos'] = train_df['photos'].apply(len)
train_df = train_df.drop('photos', axis=1)

# features numbers & high frequency features
train_df['num_features'] = train_df['features'].apply(len)

# change list of features to a long string 
features = train_df_real['features'].array
row = features.shape[0]
for i in range(row):
    temp = features[i]
    string = ''
    for j in range(len(temp)):
        string += temp[j].replace(" ", "_")
        string += " "
    features[i] = string

# vectorize the features
context = features
for i in range(len(context)):
    context[i] = func.stemSentence(context[i])

features = train_df['features'].array
row = features.shape[0]
for i in range(row):
    temp = features[i]
    string = ''
    for j in range(len(temp)):
        string += temp[j].replace(" ", "_")
        string += " "
    features[i] = string

# vectorize the features
corpus = features
for i in range(len(corpus)):
    corpus[i] = func.stemSentence(corpus[i])

vectorizer = TfidfVectorizer(min_df=0.1)
vectorizer.fit(context)
X = vectorizer.transform(corpus)
df_description = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names(), index=train_df.axes[0][:])
# df_description['features'] = train_df['features'].array
train_df = pd.concat([train_df, df_description], axis=1)
print(train_df)
train_df = train_df.drop('features', axis=1)

# add new features -- distance from Central park
latitudeMean = 40.785091
longitudeMean = -73.968285
site_coords=(latitudeMean, longitudeMean)
train_df['distance'] =train_df.apply(lambda row:distance.distance(site_coords, (row.latitude, row.longitude)).km, axis=1)

# train_df['interest_level'] = train_df.apply
# train_df.interest_level[train_df.interest_level=='high'] = 2
# train_df.interest_level[train_df.interest_level=='medium'] = 1
# train_df.interest_level[train_df.interest_level=='low'] = 0

# price per bedroom
train_df['price_per_bed'] = train_df['price']/(train_df['bedrooms']+1)
# price per bathroom
train_df['price_per_bath'] = train_df['price']/(train_df['bathrooms']+1)
# total room
train_df['room_num'] = train_df['bathrooms']+train_df['bedrooms']
# the number of listing the manager uploaded
manager_id = train_df['manager_id'].value_counts()
train_df['manager_count'] = list(map(lambda x:manager_id[x], train_df['manager_id']))



# lbl = preprocessing.LabelEncoder()
# list1 = list(train_df_real['manager_id'].values)
# print(len(set(list1)))
# list2 = list(train_df['manager_id'].values)
# print(len(set(list2)))
# list3 = list(set(list1 + list2))
# print(len(set(list3)))
# lbl.fit(list3)
# train_df_real['manager_id'] = lbl.transform(list1)
# # features = train_df.axes[1][:]
# # p = features.index('interest_level')
# # train_values = train_df.values[:, :]
# y = train_df_real['interest_level']
# temp = pd.concat([train_df_real.manager_id, pd.get_dummies(y)], axis=1).groupby('manager_id').mean()
# temp.columns = ['low_frac', 'medium_frac', 'high_frac']
# temp['count'] = train_df_real.groupby('manager_id').count().iloc[:,1]
# temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
# mean = temp.loc[temp['count'] >= 10, 'manager_skill'].mean()
# temp.loc[temp['count'] < 10, 'manager_skill'] = mean
# # print(temp)
# train_df['manager_id'] = lbl.transform(list2)
# train_df = pd.merge(left=train_df, right=temp[['manager_skill']], how='left', left_on='manager_id', right_index=True)
# print(train_df.head())
# # train_df['manager_skill'] = pd.to_numeric(train_df['manager_skill'], errors='coerce')
# train_df['manager_skill'].fillna(mean, inplace=True)
# print(train_df['manager_skill'])


train_df = train_df.drop('manager_id', axis=1)
# bedroom percentage
train_df['bedsPerc'] = train_df['bedrooms']/(train_df['bedrooms']+train_df['bathrooms']+1)
# bathrooms percentage
train_df['bathPerc'] = train_df['bathrooms']/(train_df['bedrooms']+train_df['bathrooms']+1)


print(train_df.shape)
features = train_df_final.axes[1][:]
features_test = train_df.axes[1][:]
for feature in features_test:
    if feature not in features:
        train_df = train_df.drop(feature, axis=1)
print(features)

train_df.to_json('new_test.json')