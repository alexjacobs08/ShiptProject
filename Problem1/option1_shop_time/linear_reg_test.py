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
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree

datafile='/Users/alexanderjacobs/Repos/ShiptProjects/Problem1/option1_shop_time/shipt_takehome_shop_time.tsv'

def preprocess(data, test_size, sample=None, scale=True):

    data_frame_all = pandas.read_table(data)
    df = data_frame_all

    # for simplicity for now--and since only 11093 or <3 % of our data, we're just gonna drop those rows
    no_null_df = df.dropna(axis=0, how='any')

    # this shows us that we no longer have null values
    no_null_df.isnull().values.any()

    # let's rename our new data frame df again.  we're left with 238907 rows
    df = no_null_df

    obj_df = df.select_dtypes(include=['object']).copy()

    # changes type to category
    for header in list(obj_df):
        obj_df[header] = obj_df[header].astype('category')

    # creates a list of headers + "_cat"  this could be done inplace
    cat_header_list = []
    for header in list(obj_df):
        cat_header_list.append(header + "_cat")

    # adds three columbs to df which have the data coded as a category

    for header in list(obj_df):
        new_header_name = header + "_cat"
        obj_df[new_header_name] = obj_df[header].cat.codes

    # replaces object columns in df with their categorical equivalent
    for i in range(len(cat_header_list)):
        df[list(obj_df)[i]] = obj_df[cat_header_list[i]]


    if sample:
        df = df.sample(frac=sample)
        print("sampled")


    items_in_order = df[['cat_dairy', 'cat_meat',
     'cat_produce', 'cat_beverages', 'cat_breads', 'cat_dry_goods', 'cat_accessories', 'cat_baby', 'cat_breakfast',
     'cat_canned_goods', 'cat_coffee_tea', 'cat_deli', 'cat_bakery', 'cat_frozen', 'cat_household', 'cat_international',
     'cat_pantry', 'cat_personal_care', 'cat_snacks', 'cat_other_unknown']]

    should_be_one = df["order_num_order_lines"] / items_in_order


    for category in list(items_in_order):

        newdf = df[category] / df["order_num_order_lines"]

        df[category] = newdf





    # assign our target value to target
    # drop the target column from our dataframe
    target = df["actual_shopping_duration_min"]
    df = df.drop(["order_id", "order_delivery_hour","order_time_to_delivery_min","order_delivery_month","metro_id","store_id", "shopper_id", "shopper_yob", "member_substitution_preference", "order_delivery_month", "actual_shopping_duration_min"], axis=1)

    # ['order_num_order_lines', 'order_num_special_requests', 'order_time_to_delivery_min', 'order_delivery_hour',
    #  'order_delivery_dow', 'order_delivery_month', 'member_substitution_preference', 'cat_dairy', 'cat_meat',
    #  'cat_produce', 'cat_beverages', 'cat_breads', 'cat_dry_goods', 'cat_accessories', 'cat_baby', 'cat_breakfast',
    #  'cat_canned_goods', 'cat_coffee_tea', 'cat_deli', 'cat_bakery', 'cat_frozen', 'cat_household', 'cat_international',
    #  'cat_pantry', 'cat_personal_care', 'cat_snacks', 'cat_other_unknown', 'store_pick_rate_min', 'metro_id',
    #  'shopper_num_prev_shops', 'shopper_num_prev_shops_at_store', 'shopper_pick_rate_min', 'shopper_age',
    #  'shopper_gender']

    X = df
    y = target

    if scale:
        df_pp = preprocessing.scale(df)
        print("scaled")

        X_train, X_test, y_train, y_test = train_test_split(df_pp, target, test_size=test_size, random_state=42)

    else:
        df_pp = None
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)


    return df, df_pp, obj_df, target, X, X_train, X_test, y, y_train, y_test


df, df_pp, obj_df, target, X, X_train, X_test, y, y_train, y_test = preprocess(datafile, test_size=.33, sample=.1, scale=False)

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier


# ser, bins = pandas.qcut(y,10,retbins=True)
#
# print(type(ser))
# print(bins)
bin_means = (numpy.histogram(y, 30))#[0] / numpy.histogram(y, 10)[0])

#print(bin_means)

# print(X)
# print(y_tree)
#
# model = ExtraTreesClassifier()
# model.fit(X, y_tree)
# print(model.feature_importances_)

#print(y_train.shape)


selector = SelectKBest(f_regression, k=10).fit(X_train, y_train)
#print(Xbest.scores_)
#print(Xbest.pvalues_)
X_new = selector.fit_transform(X_train, y_train)


# print("training linear model")
# lm = linear_model.LinearRegression()
# model = lm.fit(X_train, y_train)
#
# print("predictions")
# predictions = lm.predict(X_train)
# score = lm.score(X_train, y_train)

pf = PolynomialFeatures(degree=3,interaction_only=True)
poly_X_train = pf.fit_transform(X_new)



lm = linear_model.LinearRegression()
model = lm.fit(poly_X_train,y_train)

predictions = lm.predict(poly_X_train)
score = lm.score(poly_X_train, y_train)


numpy.mean(predictions - y_train)

print(score)
pred_new = predictions / 1.7
print(numpy.mean(predictions/y_train))

print(numpy.mean(abs(predictions - y_train)))


print("#####################")

regr_1 = tree.DecisionTreeRegressor(max_depth=500)
regr_1.fit(X_train, y_train)
predictions = regr_1.predict(X_test)

min_error = numpy.mean(abs(predictions - y_test))

print(min_error)

# score_list = []
# rfe_list = []
# minutes_off_avg = []
#
# start = time.time()
# # for feature_num in range(X.shape[1],3,-1):
# for feature_num in range(5,4,-1):
#
#     lm = sklearn.linear_model.LinearRegression()
#     model = lm.fit(X_train, y_train)
#
#     rfe = RFE(model, feature_num)
#     rfe = rfe.fit(X_train, y_train)
#
#     rfe_ind = [i for i, x in enumerate(rfe.support_) if x]
#     rfe_list.append(rfe_ind)
#
#     X_rfe = df.ix[:, rfe_ind]
#
#     model = lm.fit(X_rfe, y_train)
#
#     score = model.score(X_rfe,y_train)
#
#     predictions = lm.predict(X_rfe)
#
#     minutes_off_avg.append(numpy.mean(predictions - target))
#
#     score_list.append([feature_num, score])
#
#
# end = time.time()
#
# print(end-start)
#
# print(score_list)
#
# print(rfe_list)
#
# print(minutes_off_avg)
#
#




