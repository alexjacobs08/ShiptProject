import csv
import pandas
import numpy
import matplotlib
matplotlib.use('TkAgg')
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
import statsmodels
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn

datatsv = '/Users/alexanderjacobs/Repos/ShiptProjects/Problem1/option1_shop_time/shipt_takehome_shop_time.tsv'

data_frame_all = pandas.read_table(datatsv)
df = data_frame_all

#for simplicity for now--and since only 11093 or <3 % of our data, we're just gonna drop those rows
no_null_df = df.dropna(axis=0, how='any')

#this shows us that we no longer have null values
no_null_df.isnull().values.any()

#let's rename our new data frame df again.  we're left with 238907 rows
df = no_null_df

obj_df = df.select_dtypes(include=['object']).copy()

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
df = df.sample(frac=.1)
target = df["actual_shopping_duration_min"]
df.drop("actual_shopping_duration_min", axis=1, inplace=True)

X = preprocessing.scale(df)
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

lm = sklearn.linear_model.LinearRegression()
model = lm.fit(X,y)

#recursive feature elimination should remove features that aren't helping or are hurting our model
from sklearn.feature_selection import RFE



score_list = []
rfe_list = []
minutes_off_avg = []

start = time.time()
# for feature_num in range(X.shape[1],3,-1):
for feature_num in range(5,4,-1):

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

    model = lm.fit(X, y)

    rfe = RFE(model, feature_num)
    rfe = rfe.fit(X, y)

    rfe_ind = [i for i, x in enumerate(rfe.support_) if x]
    rfe_list.append(rfe_ind)

    X_rfe = df.ix[:, rfe_ind]

    model = lm.fit(X_rfe, y)

    score = model.score(X_rfe,y)

    predictions = lm.predict(X_rfe)

    minutes_off_avg.append(numpy.mean(predictions - target))

    score_list.append([feature_num, score])


end = time.time()

print(end-start)

print(score_list)

print(rfe_list)

print(minutes_off_avg)

#[0, 1, 3, 4, 5, 6, 7, 9, 12, 13, 15, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37]
# from sklearn.preprocessing import PolynomialFeatures
# start = time.time()
# # for feature_num in range(X.shape[1],3,-1):
# for feature_num in range(5, 4, -1):
#     print("enter loop")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
#     #print(X_train.shape, y_train.shape)
#
#     pf = PolynomialFeatures(degree=2)
#     poly_X_train = pf.fit_transform(X_train)
#     poly_X_test = pf.fit_transform(X_test)
#     print("pn transformation applied")
#     model = lm.fit(X_train, y_train)
#
#     rfe = RFE(model, feature_num)
#     rfe = rfe.fit(poly_X_train, y_train)
#     print("starting rfe")
#     rfe_ind = [i for i, x in enumerate(rfe.support_) if x]
#     rfe_list.append(rfe_ind)
#     print("rfe finished")
#     X_rfe = df.ix[:, rfe_ind] #only works in pandas so not with train/test split
#
#     X_train_rfe = poly_X_train[:, rfe_ind]
#     X_test_rfe = poly_X_test[:, rfe_ind]
#
#     #print(X_train_rfe.shape, y_train.shape)
#     print("starting fit model")
#     model = lm.fit(X_train_rfe, y_train)
#     #print("X_test_rfe, y_test", X_test_rfe.shape, y_test.shape)
#     score = model.score(X_test_rfe, y_test)
#
#     predictions = lm.predict(X_test_rfe)
#
#     minutes_off_avg.append(numpy.mean(predictions - y_test))
#
#     score_list.append([feature_num, score])
#
# end = time.time()
#
# print(end - start)
#
# print(score_list)
#
# print(rfe_list)
#
# print(minutes_off_avg)
