import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func
import time
import seaborn as sns

(listing_id, features, values) = func.load_unicef_data()
# print(features)
# print(values)
train_df = pd.read_json('train.json')

# hist of price
p = features.index('price') + 1
min_price = min(train_df['price'])
max_price = int(np.percentile(train_df['price'], 99))
for i in range(values[:, p].shape[0]):
    if values[i, p] > max_price:
        values[i, p] = max_price
bins_price = range(min_price, max_price, 200)

fig1 = plt.figure(num='fig1')
plt.figure(num='fig1')
plt.subplot(121)
plt.scatter(range(train_df['price'].shape[0]), np.sort(train_df['price']))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.subplot(122)
(counts, bins, patches) = plt.hist(values[:, p], bins_price, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('price', fontsize=12)
plt.ylabel('counts', fontsize=12)

# hist of latitude
p = features.index('latitude') + 1
min_latitude = np.percentile(train_df['latitude'], 1)
max_latitude = np.percentile(train_df['latitude'], 99)
for i in range(values[:, p].shape[0]):
    if values[i, p] > max_latitude:
        values[i, p] = max_latitude
    elif values[i, p] < min_latitude:
        values[i, p] = min_latitude
bins_latitude = np.linspace(min_latitude, max_latitude, 50)

fig2 = plt.figure(num='fig2')
plt.figure(num='fig2')
plt.subplot(121)
plt.scatter(range(train_df['latitude'].shape[0]), np.sort(train_df['latitude']))
plt.xlabel('index', fontsize=12)
plt.ylabel('latitude', fontsize=12)
plt.subplot(122)
(counts, bins, patches) = plt.hist(values[:, p], bins_latitude, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('latitude', fontsize=12)
plt.ylabel('counts', fontsize=12)

# hist of longitude
p = features.index('longitude') + 1
min_longitude = np.percentile(train_df['longitude'], 1)
max_longitude = np.percentile(train_df['longitude'], 99)
for i in range(values[:, p].shape[0]):
    if values[i, p] > max_longitude:
        values[i, p] = max_longitude
    elif values[i, p] < min_longitude:
        values[i, p] = min_longitude
bins_longitude = np.linspace(min_longitude, max_longitude, 50)

fig3 = plt.figure(num='fig3')
plt.figure(num='fig3')
plt.subplot(121)
plt.scatter(range(train_df['longitude'].shape[0]), np.sort(train_df['longitude']))
plt.xlabel('index', fontsize=12)
plt.ylabel('longitude', fontsize=12)
plt.subplot(122)
(counts, bins, patches) = plt.hist(values[:, p], bins_longitude, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('longitude', fontsize=12)
plt.ylabel('counts', fontsize=12)

# hour-wise listing trend
p = features.index('created') + 1
timeArray = list()
hours = list()
for value in values[:, p]:
    timeArray.append(time.strptime(value, "%Y-%m-%d %H:%M:%S"))
for time in timeArray:
    hours.append(time.tm_hour)

fig4 = plt.figure(num='fig4')
plt.figure(num='fig4')
plt.hist(hours, range(25), facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('hours', fontsize=12)
plt.ylabel('counts', fontsize=12)

# proportion of target variable values
p = features.index('interest_level') + 1
high = 0
medium = 0
low = 0
for value in values[:, p]:
    if value == "high":
        high = high+1
    elif value == "medium":
        medium = medium+1
    elif value == "low":
        low = low+1
high = high/(values[:, p].shape[0])
medium = medium/(values[:, p].shape[0])
low = low/(values[:, p].shape[0])

fig5 = plt.figure(num='fig5')
plt.figure(num='fig5')
plt.bar(['low', 'medium', 'high'], [low, medium, high])
plt.xlabel('interest_level', fontsize=12)
plt.ylabel('percentage', fontsize=12)
plt.show()
