import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import basic_func as func

# ---------missing values--------------
# building_id '0' 8286 ignore
# created  "" 0
# description "" 1446 ignore it/None
# display_address "" 135
# features [] 3218 ignore
# latitude number 0
# listing_id number 0
# longitude number 0
# manager_id str 0
# photos [] 3615 ignore
# price number 0
# street_address "" 10 drop
# interest_level 0

(listing_id, features, values) = func.load_unicef_data()
train_df = pd.read_json("train.json")
# record the number of missing values
# initialize the counts
# counts = dict()
# for feature in features:
#     counts[feature] = 0
# # print(type(values[1, p]) is int)
# print(train_df.isnull().sum())
# for p in range(0, len(features)):
#     for i in range(values[:, p].shape[0]):
#         # missing value of string types
#         if type(values[1, p]) is str:
#             if values[i, p] == "" or values[i, p].replace(" ", "") == "":
#                 values[i, p] = ""
#                 counts[features[p]] = counts[features[p]] + 1
#         # missing value of list value
#         if type(values[1, p]) is list:
#             if not values[i, p]:
#                 counts[features[p]] = counts[features[p]] + 1
# print(counts)

# handle missing value in street_address and display_address
p = features.index("display_address")
q = features.index("street_address")
disp_addr = values[:, p]
street_addr = values[:, q]
count = 0
count1 = 0
i = 0;
m_num = disp_addr.shape[0]
while True:
    if i >= values.shape[0]-1:
        break;
    # display_address is ""  & street address is "", drop the tuple.
    if values[i, p] == "" and values[i, q] == "":
        count1 += 1
        values = np.delete(values, i, axis=0)
        i -= 1
        continue;
    # display_address is "" street address is not  "", assign street address to display address without number
    elif values[i, p] == "" and values[i, q] != "":
        count = count + 1
        temp_str = values[i, q]
        for j in range(len(temp_str)):
            if temp_str[j] == " ":
                values[i, p] = temp_str[j + 1:]
                # print(temp_str + "------>  " + values[i, p])
                break
    elif values[i, q] == "" and values[i, p] != "":
        values[i, q] = values[i, p]
    i += 1
print(count1)

# outliers for description
q = features.index("description")
des = values[:, p]
length_des = list()
for i in range(des.shape[0]):
    length_des.append(len(values[i, q]))
fig_des = plt.figure(num='fig_des')
plt.figure(num='fig_des')
plt.scatter(range(len(length_des)), np.sort(length_des))
plt.xlabel('index', fontsize=12)
plt.ylabel('description length', fontsize=12)
plt.show()

# outlier display_address
a = features.index("display_address")
display = values[:, a]
length_disp_addr = list()
count2 = 0
for i in range(display.shape[0]):
    if len(values[i,a])>=55:
        count2 += 1
        print(values[i,a] + "     ")
    length_disp_addr.append(len(values[i, a]))
fig_disp_addr = plt.figure(num='fig_disp_addr')
plt.figure(num='fig_disp_addr')
plt.scatter(range(len(length_disp_addr)), np.sort(length_disp_addr))
plt.xlabel('index', fontsize=12)
plt.ylabel('display_address length', fontsize=12)
plt.show()
print(count2)
# outlier street_address
# p = features.index("street_address")
# street = values[:, p]
# length_street_addr = list()
# for i in range(street.shape[0]):
#     if len(values[i,p])>=70:
#         print(values[i,p] + "     ")
#     length_street_addr.append(len(values[i, p]))
# fig_street_addr = plt.figure(num='fig_street_addr')
# plt.figure(num='fig_street_addr')
# plt.scatter(range(len(length_street_addr)), np.sort(length_street_addr))
# plt.xlabel('index', fontsize=12)
# plt.ylabel('street_address length', fontsize=12)
# plt.show()