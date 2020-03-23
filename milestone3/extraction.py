import pandas as pd
import numpy as np
from geopy import distance
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from tkinter import _flatten
import time
import nltk
import basic_func as func
# nltk.download('vader_lexicon')

train_df = pd.read_json('train2.json')
test_df = pd.read_json('test.json')

train_df.interest_level[train_df.interest_level=='high'] = 2.0
train_df.interest_level[train_df.interest_level=='medium'] = 1.0
train_df.interest_level[train_df.interest_level=='low'] = 0.0

# created ---> month day hours
train_df['created'] = pd.to_datetime(train_df['created'])
test_df['created'] = pd.to_datetime(test_df['created'])

train_df['created_month'] = train_df['created'].dt.month
test_df['created_month'] = test_df['created'].dt.month

train_df['created_day'] = train_df['created'].dt.day
test_df['created_day'] = test_df['created'].dt.day

train_df['created_hour'] = train_df['created'].dt.hour
test_df['created_hour'] = test_df['created'].dt.hour

train_df = train_df.drop('created', axis=1)
test_df = test_df.drop('created', axis=1)

# decription count words
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))


# display address & street_address 
train_df['disp_east'] = train_df['display_address'].apply(lambda x: x.find('east')>-1).astype(int)
train_df['disp_west'] = train_df['display_address'].apply(lambda x: x.find('west')>-1).astype(int)
train_df['disp_north'] = train_df['display_address'].apply(lambda x: x.find('north')>-1).astype(int)
train_df['disp_sorth'] = train_df['display_address'].apply(lambda x: x.find('south')>-1).astype(int)

test_df['disp_east'] = test_df['display_address'].apply(lambda x: x.find('east')>-1).astype(int)
test_df['disp_west'] = test_df['display_address'].apply(lambda x: x.find('west')>-1).astype(int)
test_df['disp_north'] = test_df['display_address'].apply(lambda x: x.find('north')>-1).astype(int)
test_df['disp_sorth'] = test_df['display_address'].apply(lambda x: x.find('south')>-1).astype(int)

train_df = train_df.drop(['display_address', 'street_address'], axis=1)
test_df = test_df.drop(['display_address', 'street_address'], axis=1)

# photos numbers
train_df['num_photos'] = train_df['photos'].apply(len)
train_df = train_df.drop('photos', axis=1)

test_df['num_photos'] = test_df['photos'].apply(len)
test_df = test_df.drop('photos', axis=1)

# features numbers & high frequency features
train_df['num_features'] = train_df['features'].apply(len)
test_df['num_features'] = test_df['features'].apply(len)

# bows = {'nofee': ['no fee', 'no-fee', 'no  fee', 'nofee', 'no_fee'],
#         'lowfee': ['reduced_fee', 'low_fee', 'reduced fee', 'low fee'],
#         'furnished': ['furnished'],
#         'parquet': ['parquet', 'hardwood'],
#         'concierge': ['concierge', 'doorman', 'housekeep', 'in_super'],
#         'prewar': ['prewar', 'pre_war', 'pre war', 'pre-war'],
#         'laundry': ['laundry', 'lndry'],
#         'health': ['health', 'gym', 'fitness', 'training'],
#         'transport': ['train', 'subway', 'transport'],
#         'parking': ['parking'],
#         'utilities': ['utilities', 'heat water', 'water included']
#         }
# for fname, bow in bows.items():
#     x1 = train_df.description.str.lower().apply(lambda x: np.sum([1 for i in bow if i in x]))
#     x2 = train_df.features.apply(lambda x: np.sum([1 for i in bow if i in ' '.join(x).lower()]))
#     train_df['num_' + fname] = ((x1 + x2) > 0).astype(float).values
#     x1 = test_df.description.str.lower().apply(lambda x: np.sum([1 for i in bow if i in x]))
#     x2 = test_df.features.apply(lambda x: np.sum([1 for i in bow if i in ' '.join(x).lower()]))
#     test_df['num_' + fname] = ((x1 + x2) > 0).astype(float).values

# add new features -- distance from Central park
latitudeMean = 40.785091
longitudeMean = -73.968285
site_coords=(latitudeMean, longitudeMean)
train_df['distance'] =train_df.apply(lambda row:distance.distance(site_coords, (row.latitude, row.longitude)).km, axis=1)
test_df['distance'] =test_df.apply(lambda row:distance.distance(site_coords, (row.latitude, row.longitude)).km, axis=1)

# give the price bins
bins = train_df.price.quantile(np.arange(0.05, 0.95, 0.1))
train_df['num_price_q'] = np.digitize(train_df.price, bins)
test_df['num_price_q'] = np.digitize(test_df.price, bins)

# price per bedroom
train_df['price_per_bed'] = train_df['price']/(train_df['bedrooms']+1)
test_df['price_per_bed'] = test_df['price']/(test_df['bedrooms']+1)

# price per bathroom
train_df['price_per_bath'] = train_df['price']/(train_df['bathrooms']+1)
test_df['price_per_bath'] = test_df['price']/(test_df['bathrooms']+1)

# total room
train_df['room_num'] = train_df['bathrooms']+train_df['bedrooms']
test_df['room_num'] = test_df['bathrooms']+test_df['bedrooms']

# bedroom percentage
train_df['bedsPerc'] = train_df['bedrooms']/(train_df['bedrooms']+train_df['bathrooms']+1)
test_df['bedsPerc'] = test_df['bedrooms']/(test_df['bedrooms']+test_df['bathrooms']+1)

# bathrooms percentage
train_df['bathPerc'] = train_df['bathrooms']/(train_df['bedrooms']+train_df['bathrooms']+1)
test_df['bathPerc'] = test_df['bathrooms']/(test_df['bedrooms']+test_df['bathrooms']+1)

precision = 3
n_min = 50

# the average rating of a ceratin building id
train_df['interest_level'] = train_df.interest_level.astype(int)
impute = train_df.interest_level.mean()
x = train_df.groupby('building_id')['interest_level'].aggregate(['count', 'mean'])
d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
train_df['num_building_id'] = train_df.building_id.apply(lambda x: d.get(x, impute))
test_df['num_building_id'] = test_df.building_id.apply(lambda x: d.get(x, impute))

# Building frequency
d = np.log1p(train_df.building_id.value_counts()).to_dict()
impute = np.min(np.array(list(d.values())))
train_df['building_freq'] = train_df.building_id.apply(lambda x: d.get(x, impute))
test_df['building_freq'] = test_df.building_id.apply(lambda x: d.get(x, impute))
# print(train_df.loc[train_df['building_id']=='0'])

# building id drop
train_df = train_df.drop('building_id', axis=1)
test_df = test_df.drop('building_id', axis=1)

# manager skill
lbl = LabelEncoder()
train_manager_id = list(train_df['manager_id'].values)
test_manager_id = list(test_df['manager_id'].values)
total_manager_id = list(set(train_manager_id+test_manager_id))
lbl.fit(total_manager_id)
train_df['manager_id'] = lbl.transform(list(train_df['manager_id'].values))
test_df['manager_id'] = lbl.transform(list(test_df['manager_id'].values))

train_y = train_df['interest_level']
temp = pd.concat([train_df.manager_id, pd.get_dummies(train_y)], axis=1).groupby('manager_id').mean()
temp.columns = ['low_frac', 'medium_frac', 'high_frac']
temp['count'] = train_df.groupby('manager_id').count().iloc[:,1]
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
mean = temp.loc[temp['count'] >= 10, 'manager_skill'].mean()
temp.loc[temp['count'] < 10, 'manager_skill'] = mean
# print(temp)

train_df = pd.merge(left=train_df, right=temp[['manager_skill']], how='left', left_on='manager_id', right_index=True)
train_df['manager_skill'].fillna(mean, inplace=True)
test_df = pd.merge(left=test_df, right=temp[['manager_skill']], how='left', left_on='manager_id', right_index=True)
test_df['manager_skill'].fillna(mean, inplace=True)

# print(train_df['manager_skill'])
# print(test_df['manager_id'])

# Manager frequency
d = np.log1p(train_df.manager_id.value_counts()).to_dict()
impute = np.min(np.array(list(d.values())))
train_df['manager_freq'] = train_df.manager_id.apply(lambda x: d.get(x, impute))
test_df['manager_freq'] = test_df.manager_id.apply(lambda x: d.get(x, impute))

# manager created average hour
train_df['created_hour'] = train_df.created_hour.astype(int)
x = train_df.groupby('manager_id')['created_hour'].aggregate(['count', 'mean'])
impute = train_df.created_hour.mean()
d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
train_df['hour_manager'] = train_df.manager_id.apply(lambda x: d.get(x, impute))
test_df['hour_manager'] = test_df.manager_id.apply(lambda x: d.get(x, impute))

# manager created average price
x = train_df.groupby('manager_id')['price'].aggregate(['count', 'mean'])
impute = train_df.price.mean()
d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
train_df['price_manager'] = train_df.manager_id.apply(lambda x: d.get(x, impute))
test_df['price_manager'] = test_df.manager_id.apply(lambda x: d.get(x, impute))


train_df = train_df.drop('manager_id', axis=1)
test_df = test_df.drop('manager_id', axis=1)

train_df['pos'] = train_df.longitude.round(precision).astype(str) + '_' + train_df.latitude.round(precision).astype(str)
test_df['pos'] = test_df.longitude.round(precision).astype(str) + '_' + test_df.latitude.round(precision).astype(str)

train_df['interest_level'] = train_df.interest_level.astype(int)
impute = train_df.interest_level.mean()

# Average interest in unique locations at given precision
x = train_df.groupby('pos')['interest_level'].aggregate(['count', 'mean'])
d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
train_df['num_pos'] = train_df.pos.apply(lambda x: d.get(x, impute))
test_df['num_pos'] = test_df.pos.apply(lambda x: d.get(x, impute))

# Density in unique locations at given precision
vals = train_df['pos'].value_counts()
dvals = vals.to_dict()
train_df['num_pos_density'] = train_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
test_df['num_pos_density'] = test_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
train_df = train_df.drop('pos', axis=1)
test_df = test_df.drop('pos', axis=1)


# # description sentiment analysis
# def description_sentiment(sentences):
#     analyzer = SentimentIntensityAnalyzer()
#     result = []
#     for sentence in sentences:
#         vs = analyzer.polarity_scores(sentence)
#         result.append(vs)
#     return pd.DataFrame(result).mean()
# train_df['description_tokens'] = train_df['description'].apply(sent_tokenize)
# train_df['description_senti'] = train_df['description_tokens'].apply(description_sentiment)
# test_df['description_tokens'] = test_df['description'].apply(sent_tokenize)
# test_df['description_senti'] = test_df['description_tokens'].apply(description_sentiment)

# train_df = train_df.drop('description_tokens', axis=1)
# test_df = test_df.drop('description_tokens', axis=1)


# print("begin sentiment")
# train_df['sentiment'] = train_df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
# test_df['sentiment'] = test_df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

# change list of features to a long string 
# features = train_df['features'].array
# row = features.shape[0]
# for i in range(row):
#     temp = features[i]
#     string = ''
#     for j in range(len(temp)):
#         string += temp[j].replace(" ", "_")
#         string += " "
#     features[i] = string

# # vectorize the features
# corpus = features
# for i in range(len(corpus)):
#     corpus[i] = func.stemSentence(corpus[i])
# vectorizer = TfidfVectorizer(min_df=0.1)
# X = vectorizer.fit_transform(corpus)
# df_description = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names(), index=train_df.axes[0])
# # df_description['features'] = train_df['features'].array
# train_df = pd.concat([train_df, df_description], axis=1)

# features = test_df['features'].array
# row = features.shape[0]
# for i in range(row):
#     temp = features[i]
#     string = ''
#     for j in range(len(temp)):
#         string += temp[j].replace(" ", "_")
#         string += " "
#     features[i] = string

# # vectorize the features
# context = features
# for i in range(len(context)):
#     context[i] = func.stemSentence(context[i])

# X = vectorizer.transform(context)
# df_description = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names(), index=test_df.axes[0][:])
# # df_description['features'] = train_df['features'].array
# test_df = pd.concat([test_df, df_description], axis=1)

train_df = train_df.drop('features', axis=1)
test_df = test_df.drop('features', axis=1)

train_df = train_df.drop('description', axis=1)
test_df = test_df.drop('description', axis=1)
# print("finish sentiment")


# generate json file
print(train_df)
print(test_df)

train_df.to_json('new_train.json')
test_df.to_json('new_test.json')

