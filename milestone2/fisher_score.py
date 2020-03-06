import pandas as pd
import numpy as np
import basic_func as func

(listing_id, features, values, data) = func.load_unicef_data("train3.json")

df = data
high = df.loc[df["interest_level"] == 2, ["listing_id"]]
df_high = df.loc[df["listing_id"].isin(high["listing_id"])]

medium = df.loc[df["interest_level"] == 1, ["listing_id"]]
df_medium = df.loc[df["listing_id"].isin(medium["listing_id"])]

low = df.loc[df["interest_level"] == 0, ["listing_id"]]
df_low = df.loc[df["listing_id"].isin(low["listing_id"])]

(listing_high, values_high) = func.load_by_data(df_high)
(listing_medium, values_medium) = func.load_by_data(df_medium)
(listing_low, values_low) = func.load_by_data(df_low)
# proportion of target variable values
p = features.index('interest_level')
# get the counts of different levels
high = 0
medium = 0
low = 0
for value in values[:, p]:
    if value == 2:
        high = high + 1
    elif value == 1:
        medium = medium + 1
    elif value == 0:
        low = low + 1
# calculate the proportion of different levels
high = high / (values[:, p].shape[0])
medium = medium / (values[:, p].shape[0])
low = low / (values[:, p].shape[0])
probability = [high, medium, low]
Fisher_score = []
mean_of_feature = []
mean_of_feature_high = []
mean_of_feature_medium = []
mean_of_feature_low = []
variance_high = []
variance_medium = []
variance_low = []

for i in range(values.shape[1] - 1):
    temp = sum(values[:, i]) / values.shape[0]
    mean_of_feature.append(temp)
for i in range(values_high.shape[1] - 1):
    temp = sum(values_high[:, i]) / values_high.shape[0]
    mean_of_feature_high.append(temp)
    variance_high.append(np.var(values_high[:, i]))
for i in range(values_medium.shape[1] - 1):
    temp = sum(values_medium[:, i]) / values_medium.shape[0]
    mean_of_feature_medium.append(temp)
    variance_medium.append(np.var(values_medium[:, i]))
for i in range(values_low.shape[1] - 1):
    temp = sum(values_low[:, i]) / values_low.shape[0]
    mean_of_feature_low.append(temp)
    variance_low.append(np.var(values_low[:, i]))

for i in range(values.shape[1] - 1):
    numerator = 0
    denominator = 0
    numerator = probability[0] * (mean_of_feature_high[i] - mean_of_feature[i]) ** 2 \
                + probability[1] * (mean_of_feature_medium[i] - mean_of_feature[i]) ** 2 \
                + probability[2] * (mean_of_feature_low[i] - mean_of_feature[i]) ** 2
    denominator = probability[0]*variance_high[i] + probability[1]*variance_medium[i] + probability[2]*variance_low[i]
    Fisher_score.append(numerator/denominator)
print(Fisher_score)
selected = []
for i in range(len(Fisher_score)):
    if(Fisher_score[i]>0.002):
        print(i)
        selected.append(i)
for i in selected:
    print(features[i])
print(len(selected))
