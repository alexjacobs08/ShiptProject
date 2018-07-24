import pandas
import numpy
import matplotlib
matplotlib.use('TkAgg')
from sklearn import linear_model
from sklearn import preprocessing
import statsmodels
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
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

    #df = df.drop(["order_id", "order_delivery_hour","order_time_to_delivery_min","order_delivery_month","metro_id","store_id", "shopper_id", "shopper_yob", "member_substitution_preference", "order_delivery_month"], axis=1)
    df = df.drop(["order_id", "order_time_to_delivery_min","metro_id", "shopper_id", "shopper_yob", "member_substitution_preference"], axis=1)

    headers = list(df)
    #headers = headers[:-1] #don't include "actual_shopping_duration_min" in outlier removal

    for header in headers:
        df = df[numpy.abs(df[header] - df[header].mean()) <= (3 * df[header].std())]

    target = df["actual_shopping_duration_min"]
    df = df.drop(["actual_shopping_duration_min"], axis=1)

    #X and Y are data before train test split
    X = df
    y = target

    #scale data if scale set to True
    if scale:
        df[['order_num_special_requests','store_pick_rate_min','shopper_num_prev_shops',
            'shopper_num_prev_shops_at_store','shopper_pick_rate_min','shopper_age']] = minmax_scale(df[[
            'order_num_special_requests','store_pick_rate_min','shopper_num_prev_shops',
            'shopper_num_prev_shops_at_store','shopper_pick_rate_min','shopper_age']])

        print("scaled")

        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=False)

    else:
        df_pp = None
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=False)


    return df, obj_df, target, X, X_train, X_test, y, y_train, y_test


df, obj_df, target, X, X_train, X_test, y, y_train, y_test = preprocess(datafile, test_size=.33, sample=.1, scale=True)


number_of_runs = 3 #how many times to run.  new test and train split each time


lm_perform = []
dtr_perform = []
svm_perform = []
baseline_perform = []


for run in range(number_of_runs):

    df, obj_df, target, X, X_train, X_test, y, y_train, y_test = preprocess(datafile, test_size=.33, sample=.1,
                                                                            scale=True)

    #selector = SelectKBest(f_regression, k=15).fit(X_train, y_train)
    #print(Xbest.scores_)
    #print(Xbest.pvalues_)
    #X_train_new = selector.fit_transform(X_train, y_train)
    #X_test_new = selector.transform(X_test)
    #
    # pf = PolynomialFeatures(degree=3,interaction_only=True)
    # poly_X_train = pf.fit_transform(X_train_new)
    # poly_X_test = pf.transform(X_test_new)
    #
    # X_train = poly_X_train
    # X_test = poly_X_test

    # poly = PolynomialFeatures(3, include_bias=False)
    # poly_X_train = poly.fit_transform(X_train)
    # poly_X_test = poly.transform(X_test)
    #
    #
    # #poly_model = make_pipeline(PolynomialFeatures(3), linear_model.Ridge(alpha=.1))
    # lm_poly = linear_model.LinearRegression(fit_intercept=True)
    # lm_poly.fit(poly_X_train,y_train)
    # predictions_lm_poly = lm_poly.predict(poly_X_test)
    #
    # print("Linear Poly Regression Model")
    # print(numpy.mean(abs(predictions_lm_poly - y_test)))
    # print(lm_poly.score(poly_X_test,y_test))

    lm = linear_model.LinearRegression(fit_intercept=True)
    lm.fit(X_train,y_train)
    predictions_lm = lm.predict(X_test)

    lm_perform.append([numpy.mean(predictions_lm - y_test), numpy.mean(abs(predictions_lm - y_test)),
                       numpy.median(predictions_lm - y_test), lm.score(X_test, y_test)])


    #svm_regression_model = svm.SVR(kernel='poly')
    # svm_regression_model.fit(X_train,y_train)
    # score = svm_regression_model.score(X_train,y_train)
    # predictions_svm = svm_regression_model.predict(X_test)

    # svm_perform.append([numpy.mean(predictions_svm - y_test),numpy.mean(abs(predictions_svm - y_test)),
    #                    numpy.median(predictions_svm - y_test),svm_regression_model.score(X_test, y_test)])

    dtr = tree.DecisionTreeRegressor(max_depth=500)
    dtr.fit(X_train, y_train)
    predictions_DTR = dtr.predict(X_test)

    dtr_perform.append([numpy.mean(predictions_DTR - y_test),numpy.mean(abs(predictions_DTR - y_test)),
                       numpy.median(predictions_DTR - y_test),dtr.score(X_test, y_test)])

    predictions_baseline = X_test.order_num_order_lines * 2.5

    baseline_perform.append([numpy.mean(predictions_baseline - y_test),numpy.mean(abs(predictions_baseline - y_test)),
                       numpy.median(predictions_baseline - y_test),'NA'])




#dataframes giving performance metrics

lm_perform_df = pandas.DataFrame(lm_perform, columns=['Average Error','Average Absolute Error','Median Error','R^2'])
svm_perform_df = pandas.DataFrame(svm_perform, columns=['Average Error','Average Absolute Error','Median Error','R^2'])
dtr_perform_df = pandas.DataFrame(dtr_perform, columns=['Average Error','Average Absolute Error','Median Error','R^2'])
baseline_perform_df = pandas.DataFrame(baseline_perform, columns=['Average Error','Average Absolute Error','Median Error','R^2'])




#Create bins, bin our target values, bin our predictions, see how many are equal


# bins = numpy.arange(y_test.min(),y_test.max(),2.5)
# target_bins = numpy.digitize(y_test,bins)
# predict_bins = numpy.digitize(predictions_lm ,bins)
# number_right = numpy.sum(target_bins == predict_bins)
#
# print(number_right)

