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
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

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
    df = df.drop(["order_id", "store_id", "shopper_id", "shopper_yob", "member_substitution_preference", "order_delivery_month", "actual_shopping_duration_min"], axis=1)

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


df, df_pp, obj_df, target, X, X_train, X_test, y, y_train, y_test = preprocess(datafile, test_size=.33, sample=None, scale=True)


# need to make data smaller
# svm_regression_model = svm.SVR(kernel='poly')
# svm_regression_model.fit(X_train,y_train)
# score = svm_regression_model.score(X_train,y_train)
#
# print(score)


selector = SelectKBest(f_regression, k=15).fit(X_train, y_train)
#print(Xbest.scores_)
#print(Xbest.pvalues_)
X_new = selector.fit_transform(X_train, y_train)
test_new = selector.fit_transform(X_test, y_test)

X_train_old = X_train
X_train = X_new

# Fit regression model
regr_1 = tree.DecisionTreeRegressor(max_depth=100)
regr_1.fit(X_train, y_train)

y_1 = regr_1.predict(test_new)
print(numpy.mean(abs(y_1-y_test)))

regr_2 = tree.DecisionTreeRegressor(max_depth=50000)

regr_2.fit(X_train, y_train)

y_2 = regr_2.predict(test_new)
print(numpy.mean(abs(y_2-y_test)))


# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue",
#          label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()

