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



    # assign our target value to target
    # drop the target column from our dataframe
    target = df["actual_shopping_duration_min"]
    df = df.drop(["order_id", "store_id", "shopper_id", "shopper_yob", "cat_other_unknown", "member_substitution_preference", "order_delivery_month", "cat_accessories", "actual_shopping_duration_min"], axis=1)

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

values = pandas.DataFrame()
values = df.order_num_order_lines * df.shopper_pick_rate_min

new_df = pandas.DataFrame()

new_df = df[["order_num_order_lines", "order_delivery_dow", "order_num_special_requests","shopper_pick_rate_min", "store_pick_rate_min"]]



pf = PolynomialFeatures(degree=6,interaction_only=True)
print("training polyomoia model")
poly_X_train = pf.fit_transform(new_df)

print("training linear model")
lm = linear_model.LinearRegression()
model = lm.fit(poly_X_train,target)

print("predictions")
predictions = lm.predict(poly_X_train)
score = lm.score(poly_X_train, target)


numpy.mean(predictions - target)

print(score)
print(numpy.mean(abs(predictions - target)))


regr_1 = tree.DecisionTreeRegressor(max_depth=1000)
regr_1.fit(new_df, target)

predictions = regr_1.predict(X_train, y_train)

min_error = numpy.mean(abs(X_train - y_train))

print(min_error)