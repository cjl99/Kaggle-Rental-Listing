import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import basic_func as func

# -----------------outlier--------------
train_df = pd.read_json("./mid_train.json")
# find listing_id is unique or not
(listing_id, features, values) = func.load_unicef_data()
# p = features.index("listing_id")
# listing = values[:, p]
# list_set = set(listing)
# print(len(list_set))
# print(listing.shape[0])

# drop outlier of high price(>20000)
print(train_df)
ulimit = 20000  # find out from figure in PART1
outlier_price = train_df.loc[train_df["price"] > ulimit, ["listing_id"]]
print("The number of outliers in price: " + str(outlier_price.shape[0]))
train_df = train_df.loc[~train_df["listing_id"].isin(outlier_price["listing_id"])]
# print(df)

# drop outlier of longitude(<1% and >99%)
llimit = np.percentile(train_df['longitude'], 1)
ulimit = np.percentile(train_df['longitude'], 99)
outlier_longitude = train_df.loc[train_df["longitude"] > ulimit, ["listing_id"]]
outlier_longitude2 = train_df.loc[train_df["longitude"] < llimit, ["listing_id"]]
print("The number of outliers in longitude: " + str(outlier_longitude.shape[0] + outlier_longitude2.shape[0]))
train_df = train_df.loc[~train_df["listing_id"].isin(outlier_longitude["listing_id"])]
train_df = train_df.loc[~train_df["listing_id"].isin(outlier_longitude2["listing_id"])]

# drop outlier of latitude(<1% and >99%)
llimit = np.percentile(train_df['latitude'], 1)
ulimit = np.percentile(train_df['latitude'], 99)
outlier_latitude = train_df.loc[train_df["latitude"] > ulimit, ["listing_id"]]
outlier_latitude2 = train_df.loc[train_df["latitude"] < llimit, ["listing_id"]]
print("The number of outliers in latitude: " + str(outlier_latitude.shape[0] + outlier_latitude2.shape[0]))
train_df = train_df.loc[~train_df["listing_id"].isin(outlier_latitude["listing_id"])]
train_df = train_df.loc[~train_df["listing_id"].isin(outlier_latitude2["listing_id"])]

# years outlier
y = features.index('created')
timeArray = list()
years = list()
for value in values[:, y]:
    timeArray.append(time.strptime(value, "%Y-%m-%d %H:%M:%S"))
for time in timeArray:
    years.append(time.tm_year)
count = 0
for i in range(len(years)):
    if years[i] != 2016:
        count = count + 1
print("The number of created not in 2016: " + str(count))  # result is 0

# bathroom outlier
fig_bed = plt.figure(num='fig_bath')
plt.figure(num='fig_bath')
plt.scatter(range(train_df['bathrooms'].shape[0]), np.sort(train_df['bathrooms']))
plt.xlabel('index', fontsize=12)
plt.ylabel('bathroom', fontsize=12)
plt.show()

# drop outlier of bathroom(<1% and >99%)
ulimit = 8  # get from figure
outlier_bathroom = train_df.loc[train_df['bathrooms'] > ulimit, ["listing_id"]]
print("The number of outliers in bathroom: " + str(outlier_bathroom.shape[0]))
train_df = train_df.loc[~train_df["listing_id"].isin(outlier_bathroom["listing_id"])]

# --------------------
# bedrooms outlier
fig_bed = plt.figure(num='fig_bed')
plt.figure(num='fig_bed')
plt.scatter(range(train_df['bedrooms'].shape[0]), np.sort(train_df['bedrooms']))
plt.xlabel('index', fontsize=12)
plt.ylabel('bedrooms', fontsize=12)
plt.show()

func.generate_new_json(listing_id, features, values, 'final_train.json')
data.to_json(filename)


# ------------------
# display_address outlier
# p = features.index("display_address")
# q = features.index("description")
# length_disp_addr = list()
# for i in range(train_df['display_address'].shape[0]):
#     length_disp_addr.append(len(values[i, p]))
# fig_disp_addr = plt.figure(num='fig_disp_addr')
# plt.figure(num='fig_disp_addr')
# plt.scatter(range(len(length_disp_addr)), np.sort(length_disp_addr))
# plt.xlabel('index', fontsize=12)
# plt.ylabel('display_address lenhgth', fontsize=12)
# plt.show()
#


# deal with missing values in street_address and display_address
# p = features.index("display_address")
# q = features.index("street_address")
# disp_addr = values[:, p]
# street_addr = values[:, q]
# count = 0
# for i in range(disp_addr.shape[0]):
#     # display_address is "" street address is "", assign street address to display address without number
#     if values[i, p] == "" and values[i, q] != "":
#         count = count + 1
#         temp_str = values[i, q]
#         for j in range(len(temp_str)):
#             if temp_str[j] == " ":
#                 values[i, p] = temp_str[j + 1:]
#                 # print(temp_str + "------>  " + values[i, p])
#                 break
#
#
# count = 0
# for i in range(street_addr.shape[0]):
#     if values[i, q] == "" and values[i, p] != "":
#         count = count + 1
#         values[i, q] = values[i, p]
#         print(values[i, p] + "------>  " + values[i, q])
# print(count)
