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
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm
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

    #samples the data
    if sample:
        df = df.sample(frac=sample)
        print("sampled")

    #all order categoies
    items_in_order = df[['cat_dairy', 'cat_meat',
     'cat_produce', 'cat_beverages', 'cat_breads', 'cat_dry_goods', 'cat_accessories', 'cat_baby', 'cat_breakfast',
     'cat_canned_goods', 'cat_coffee_tea', 'cat_deli', 'cat_bakery', 'cat_frozen', 'cat_household', 'cat_international',
     'cat_pantry', 'cat_personal_care', 'cat_snacks', 'cat_other_unknown']]


    #change the number of food per category into a percent of the total order
    for category in list(items_in_order):

        newdf = df[category] / df["order_num_order_lines"]

        df[category] = newdf

    #remove 2's (assuming those are unknown) and replace them with an ~equivalent ratio to match the rest of the data
    df['shopper_gender'].replace([2], numpy.random.choice([0,1],1,p=[.65,.35]), inplace=True)



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

    # if scale:
    #     df_pp = preprocessing.scale(df)
    #     print("scaled")
    if scale:
        df[['order_num_order_lines', 'order_num_special_requests','store_pick_rate_min','shopper_num_prev_shops',
            'shopper_num_prev_shops_at_store','shopper_pick_rate_min','shopper_age']] = minmax_scale(df[['order_num_order_lines',
            'order_num_special_requests','store_pick_rate_min','shopper_num_prev_shops',
            'shopper_num_prev_shops_at_store','shopper_pick_rate_min','shopper_age']])

        print("scaled")

        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)

    else:
        df_pp = None
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)


    return df, obj_df, target, X, X_train, X_test, y, y_train, y_test


df, obj_df, target, X, X_train, X_test, y, y_train, y_test = preprocess(datafile, test_size=.33, sample=.1, scale=True)


selector = SelectKBest(f_regression, k=20).fit(X_train, y_train)
#print(Xbest.scores_)
#print(Xbest.pvalues_)
X_train_new = selector.fit_transform(X_train, y_train)
#X_test_new = selector.fit_transform(X_test)

pf = PolynomialFeatures(degree=3,interaction_only=True)
poly_X_train = pf.fit_transform(X_train_new)
#poly_X_test = pf.fit_transform(X_test_new)

X_train = poly_X_train
#X_test = poly_X_test

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)

predictions = lm.predict(X_train)

print("Linear Regression Model")
print(numpy.mean(abs(predictions - y_train)))
print(lm.score(X_train,y_train))




# plt.scatter(X_train, y_train ,color='g')
# plt.plot(X, model.predict(X_train),color='k')
#
# plt.show()

print("")
print("")
print("DTR")
regr_1 = tree.DecisionTreeRegressor(max_depth=500)
regr_1.fit(X_train, y_train)
predictions = regr_1.predict(X_test)
print(regr_1.score(X_train,y_train))
min_error = numpy.mean(abs(predictions - y_test))

print(min_error)



svm_regression_model = svm.SVR(kernel='poly')
svm_regression_model.fit(X_train,y_train)
score = svm_regression_model.score(X_train,y_train)



# from sklearn import datasets ## imports datasets from scikit-learn
# data = datasets.load_boston()
#
# # define the data/predictors as the pre-set feature names
# df = pandas.DataFrame(data.data, columns=data.feature_names)
#
# # Put the target (housing value -- MEDV) in another DataFrame
# target = pandas.DataFrame(data.target, columns=["MEDV"])
#
# X = df
# y = target['MEDV']
# lm = linear_model.LinearRegression()
# model = lm.fit(X,y)
# predictions = lm.predict(X)
# print(predictions[0:5])
#
# print(lm.score(X,y))