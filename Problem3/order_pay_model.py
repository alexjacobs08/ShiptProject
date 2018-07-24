import pandas
import numpy
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

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
    df_unprocessed = df

    if sample:
        df = df.sample(frac=sample)
        print("sampled")

    df = df[['order_estimated_driving_time_min','order_estimated_shopping_time_min']]
    df['total_time_min'] = df.sum(axis=1)
    df['time_in_hours'] = df.total_time_min.divide(60)


    target = df.time_in_hours * 15
    df = df.drop(['time_in_hours', 'total_time_min'], axis=1)


    s1 = target.std()
    s2 = 7.5 #our chosen std deviation

    m1 = target.mean()
    m2 = 15 #our chosen mean

    target = m2 + (target - m1) * s2/s1  #scale our output to a mean of 15 and std deviation of 3



    X = df
    y = target

    if scale:
        df_pp = preprocessing.scale(df)
        print("scaled")

        X_train, X_test, y_train, y_test = train_test_split(df_pp, target, test_size=test_size, random_state=42)

    else:
        df_pp = None
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)


    return df_unprocessed, df, df_pp, target, X, X_train, X_test, y, y_train, y_test


df_unprocessed, df, df_pp, target, X, X_train, X_test, y, y_train, y_test = preprocess(datatsv, test_size=.33, sample=.1)


poly = PolynomialFeatures(5)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.transform(X_test)

lm = linear_model.LinearRegression(fit_intercept=False)
model = lm.fit(poly_X_train,y_train)

payment = lm.predict(poly_X_test)

print("Payment Model")
print(numpy.mean(abs(payment - y_test)))

