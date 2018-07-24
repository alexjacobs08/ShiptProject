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
from sklearn.pipeline import make_pipeline
from sklearn import svm

from scipy import stats

datatsv = '/Users/alexanderjacobs/Repos/ShiptProjects/Problem3/shipt_takehome_shopper_pay.tsv'

data_frame_all = pandas.read_table(datatsv)
df = data_frame_all

# for simplicity for now--and since only 11093 or <3 % of our data, we're just gonna drop those rows
no_null_df = df.dropna(axis=0, how='any')

# this shows us that we no longer have null values
no_null_df.isnull().values.any()

# let's rename our new data frame df again.  we're left with 238907 rows
df = no_null_df


def preprocess(data, test_size, sample=None, scale=True):

    data_frame_all = pandas.read_table(data)
    df = data_frame_all

    # for simplicity for now--and since only 11093 or <3 % of our data, we're just gonna drop those rows
    no_null_df = df.dropna(axis=0, how='any')

    # this shows us that we no longer have null values
    no_null_df.isnull().values.any()

    # let's rename our new data frame df again.  we're left with 238907 rows
    df = no_null_df


    if sample:
        df = df.sample(frac=sample)
        print("sampled")

    df = df[['order_estimated_driving_time_min','order_estimated_shopping_time_min']]
    df['total_time_min'] = df.sum(axis=1)
    df['time_in_hours'] = df.total_time_min.divide(60)

    outliers = stats.zscore(df[['order_estimated_driving_time_min','order_estimated_shopping_time_min']])

    newdf = df[numpy.abs(df.order_estimated_driving_time_min - df.order_estimated_driving_time_min.mean()) <= (3 * df.order_estimated_driving_time_min.std()) ]

    newdf = newdf[numpy.abs(df.order_estimated_shopping_time_min - df.order_estimated_shopping_time_min.mean()) <= (3 * df.order_estimated_shopping_time_min.std()) ]
    #or
    #   numpy.abs(df.order_estimated_shopping_time_min - df.order_estimated_shopping_time_min.mean()) <= (3 * df.order_estimated_shopping_time_min.std())]

    #new_df = df[numpy.abs(df.order_estimated_driving_time_min - df.order_estimated_driving_time_min.mean()) <= (3 * df.order_estimated_driving_time_min.std())]

    df = newdf

    X = df
    target = df.time_in_hours * 15
    df = df.drop(['time_in_hours', 'order_estimated_shopping_time_min', 'order_estimated_driving_time_min'], axis=1)

    s1 = target.std()
    s2 = 5 #our chosen std deviation

    m1 = target.mean()
    m2 = 15 #out chosen mean

    target = m2 + (target - m1) * s2/s1  #scale our output to a mean of 15 and std deviation of 3



    #X = df
    #target = pandas.Series([15] * df.shape[0])
    y = target

    if scale:
        df_pp = preprocessing.scale(df)
        print("scaled")

        X_train, X_test, y_train, y_test = train_test_split(df_pp, target, test_size=test_size, random_state=42)

    else:
        df_pp = None
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)


    return newdf, df, df_pp, target, X, X_train, X_test, y, y_train, y_test


newdf, df, df_pp, target, X, X_train, X_test, y, y_train, y_test = preprocess(datatsv, test_size=.33, sample=.1, scale=False)

#prediction = numpy.mean(abs(df['order_estimated_shopping_time_min'] - df['order_actual_shopping_time_min']))

poly = PolynomialFeatures(1, include_bias=False)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.transform(X_test)

lm = linear_model.Ridge(fit_intercept=False)
model = lm.fit(poly_X_train,y_train)

payment = lm.predict(poly_X_test)

print("Payment Model")
print(numpy.mean(abs(payment - y_test)))






payment = pandas.DataFrame(payment)