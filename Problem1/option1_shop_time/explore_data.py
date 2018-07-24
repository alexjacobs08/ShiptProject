import csv
import pandas
import matplotlib
matplotlib.use('TkAgg')
import sklearn
from sklearn import linear_model
import statsmodels
import time

datatsv = '/Users/alexanderjacobs/Repos/ShiptProjects/Problem1/option1_shop_time/shipt_takehome_shop_time.tsv'


#this allows me to access data using a row['value'] format
# with open(datatsv) as tsvfile:
#   reader = csv.DictReader(tsvfile, dialect='excel-tab')
#   for row in reader:
#     print(row)
#     break
#
#
# with open(datatsv) as tsvfile:
#   reader = csv.reader(tsvfile, delimiter='\t')
#   for row in reader:
#     print(row)
#     break


data_frame_all = pandas.read_table(datatsv)
df = data_frame_all
df_5 = data_frame_all[:5]
df_5_sample = data_frame_all.sample(5)

df.shape

#check for null values
df.isnull().any()

#we have some null values

# order_id                           False
# order_num_order_lines              False
# order_num_special_requests         False
# order_time_to_delivery_min         False
# order_delivery_hour                False
# order_delivery_dow                 False
# order_delivery_month               False
# member_substitution_preference     False
# cat_dairy                          False
# cat_meat                           False
# cat_produce                        False
# cat_beverages                      False
# cat_breads                         False
# cat_dry_goods                      False
# cat_accessories                    False
# cat_baby                           False
# cat_breakfast                      False
# cat_canned_goods                   False
# cat_coffee_tea                     False
# cat_deli                           False
# cat_bakery                         False
# cat_frozen                         False
# cat_household                      False
# cat_international                  False
# cat_pantry                         False
# cat_personal_care                  False
# cat_snacks                         False
# cat_other_unknown                  False
# store_id                           False
# store_pick_rate_min                False
# metro_id                           False
# shopper_id                          True
# shopper_num_prev_shops              True
# shopper_num_prev_shops_at_store     True
# shopper_pick_rate_min               True
# shopper_yob                         True
# shopper_age                         True
# shopper_gender                     False
# actual_shopping_duration_min        True



#for simplicity for now--and since only 11093 or <3 % of our data, we're just gonna drop those rows
no_null_df = df.dropna(axis=0, how='any')

#this shows us that we no longer have null values
no_null_df.isnull().values.any()

#let's rename our new data frame df again.  we're left with 238907 rows
df = no_null_df


#this will drop all columns with a NaN
#df.dropna(axis=1, how='any')

#this will drop all rows with an NaN value
#df.dropna(axis=0, how='any')

df.dtypes

obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

obj_df["order_delivery_month"].value_counts()
# August       38947
# July         34231
# May          29978
# June         29869
# April        26938
# March        25518
# February     24589
# January      21585
# September     7252


obj_df["member_substitution_preference"].value_counts()
# contact_me           203249
# use_best_judgment     29347
# do_not_substitute      3712
# unknown                2599

obj_df["shopper_gender"].value_counts()
#F          161108
#M           72062
#unknown      5737


#changes type to category
for header in list(obj_df):
    obj_df[header] = obj_df[header].astype('category')


#creates a list of headers + "_cat"  this could be done inplace
cat_header_list = []
for header in list(obj_df):
    cat_header_list.append(header + "_cat")

#adds three columbs to df which have the data coded as a category

for header in list(obj_df):
    new_header_name = header + "_cat"
    obj_df[new_header_name] = obj_df[header].cat.codes

#replaces object columns in df with their categorical equivalent
for i in range(len(cat_header_list)):
    df[list(obj_df)[i]] = obj_df[cat_header_list[i]]



#assign our target value to target
#drop the target column from our dataframe
target = df["actual_shopping_duration_min"]
df.drop("actual_shopping_duration_min", axis=1, inplace=True)

X = df
y = target



lm = sklearn.linear_model.LinearRegression()
model = lm.fit(X,y)


#recursive feature elimination should remove features that aren't helping or are hurting our model
from sklearn.feature_selection import RFE

rfe = RFE(model, 6)
rfe = rfe.fit(X, y)


rfe_ind = [i for i, x in enumerate(rfe.support_) if x]
X_rfe = df.ix[:, rfe_ind]

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

predictions = lm.predict(X)
print(predictions)[0:5]


#R^2 score
lm.score(X_rfe,y)

score_list = []
rfe_list = []

start = time.time()
for feature_num in range(X.shape[1],3,-1):

    model = lm.fit(X, y)

    rfe = RFE(model, feature_num)
    rfe = rfe.fit(X, y)

    rfe_ind = [i for i, x in enumerate(rfe.support_) if x]
    rfe_list.append(rfe_ind)
    X_rfe = df.ix[:, rfe_ind]

    score = lm.score(X_rfe,y)

    score_list.append([feature_num, score])



end = time.time()

print(end-start)

#############
## SKlearn NN


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

mlp.fit(X_train,y_train)

#from pandas.plotting import scatter_matrix
#scatter_matrix(data_frame_all.sample(5000), alpha=0.2, figsize=(6, 6), diagonal='kde')