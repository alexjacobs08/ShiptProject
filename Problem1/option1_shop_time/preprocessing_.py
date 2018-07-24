import pandas
from sklearn.model_selection import train_test_split

datatsv = '/Users/alexanderjacobs/Repos/ShiptProjects/Problem1/option1_shop_time/shipt_takehome_shop_time.tsv'
test_size = .33
#data = datatsv


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
target = df["actual_shopping_duration_min"]
df.drop("actual_shopping_duration_min", axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)

def preprocess(data, test_size):
    datatsv = data
    data_frame_all = pandas.read_table(datatsv)
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

    # assign our target value to target
    # drop the target column from our dataframe
    target = df["actual_shopping_duration_min"]
    df.drop("order_id","store_id","shopper_id","shopper_yob","actual_shopping_duration_min", axis=1, inplace=True)



    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=42)