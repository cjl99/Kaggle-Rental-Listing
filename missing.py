import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func

(listing_id, features, values, train_df) = func.load_unicef_data("train.json")

# record the number of missing values
# initialize the counts
counts = dict()
for feature in features:
    counts[feature] = 0
# print(type(values[1, p]) is int)
print(train_df.isnull().sum())
for p in range(0, len(features)):
    for i in range(values[:, p].shape[0]):
        # missing value of string types
        if type(values[1, p]) is str:
            if values[i, p] == "" or values[i, p].replace(" ", "") == "":
                values[i, p] = ""
                counts[features[p]] = counts[features[p]] + 1
        # missing value of list value
        if type(values[1, p]) is list:
            if not values[i, p]:
                counts[features[p]] = counts[features[p]] + 1
# count is missing value
print(counts)

# handle missing value in street_address and display_address
p = features.index("display_address")
q = features.index("street_address")
disp_addr = values[:, p]
street_addr = values[:, q]
# count1 means both missing in display_address & street_address
count1 = 0
i = 0
m_num = disp_addr.shape[0]
while True:
    if i >= values.shape[0] - 1:
        break
    # display_address is ""  & street address is "", drop the tuple.
    if values[i, p] == "" and values[i, q] == "":
        count1 += 1
        values = np.delete(values, i, axis=0)
        listing_id = np.delete(listing_id, i, axis=0)
    # display_address is "" street address is not  "", assign street address to display address without number
    elif values[i, p] == "" and values[i, q] != "":
        temp_str = values[i, q]
        for j in range(len(temp_str)):
            if temp_str[j] == " ":
                values[i, p] = temp_str[j + 1:]
                if values[i, p] == "":
                    values[i, p] = values[i, q]
                # print(temp_str + "------>  " + values[i, p])
                break
    elif values[i, q] == "" and values[i, p] != "":
        values[i, q] = values[i, p]
    i += 1
print("The number of tuples removed by display_address & street_address: " + str(count1))


# outliers for description -- ignore
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

# outlier display_address -- replaced by street_address
a = features.index("display_address")
p = features.index("street_address")
display = values[:, a]
length_disp_addr = list()
count2 = 0
for i in range(display.shape[0]):
    if len(values[i, a]) >= 55:
        count2 += 1
        # print("The outlier in display address: " + values[i, a])
        # action --replaced by street_address
        temp_str = values[i, p]
        for j in range(len(temp_str)):
            if temp_str[j] == " ":
                values[i, a] = temp_str[j + 1:]
                if values[i, a] == "":
                    values[i, a] = values[i, p]
                break
        # print("The value after replaced: " + values[i, a])
    length_disp_addr.append(len(values[i, a]))
# plot the figure
fig_disp_addr = plt.figure(num='fig_disp_addr')
plt.figure(num='fig_disp_addr')
plt.scatter(range(len(length_disp_addr)), np.sort(length_disp_addr))
plt.xlabel('index', fontsize=12)
plt.ylabel('display_address length', fontsize=12)
plt.show()
print("The number of outlier in display_address: " + str(count2))

street = values[:, p]
length_street_addr = list()
del_i = 0
for i in range(street.shape[0]):
    if len(values[i, p]) >= 100:
        del_i = i
        print("The outlier in street address: " + values[i, p])
    length_street_addr.append(len(values[i, p]))
# plot the figure
fig_street_addr = plt.figure(num='fig_street_addr')
plt.figure(num='fig_street_addr')
plt.scatter(range(len(length_street_addr)), np.sort(length_street_addr))
plt.xlabel('index', fontsize=12)
plt.ylabel('street_address length', fontsize=12)
plt.show()
# action -- delete one tuple
values = np.delete(values, del_i, axis=0)
listing_id = np.delete(listing_id, del_i, axis=0)

func.generate_new_json(listing_id, features, values, 'mid_train.json')