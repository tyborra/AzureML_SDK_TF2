import pandas as pd
import numpy as np

'''
Synthetic training data consists of three uniquely initialized categorical features and one random numeric feature.
The target is a random numeric sample.
'''

num_users = 100  # number of users to add to user_df

# Initialize df with one user
user_df = pd.DataFrame(pd.Series(np.full(120, 1)), columns=['user_id'])

# category 1
user_df['cat1'] = pd.Series(np.arange(1, 121))

# category 2
onesp = pd.Series(np.full(40, 1))
twosp = pd.Series(np.full(40, 2))
threesp = pd.Series(np.full(40, 3))
user_df['cat2'] = (pd.concat([onesp, twosp, threesp], axis=0, ignore_index=True))

# category 3
sub_cat = pd.Series(np.arange(1, 41))
user_df['cat3'] = pd.concat([sub_cat] * 3, axis=0, ignore_index=True)

# numeric feature
user_df['numeric1'] = (np.random.random_sample((120,))) * 10

# target
user_df['target'] = np.random.random_sample((120,))

for i in range(1, num_users):
    '''
    Add the remaining num_users - 1 users to df.
    '''

    user_id = i
    temp_user_df = pd.DataFrame(pd.Series(np.full(120, user_id)), columns=['user_id'])
    temp_user_df['cat1'] = np.random.randint(1, 121, size=(120, 1))

    onesp = pd.Series(np.full(40, 1))
    twosp = pd.Series(np.full(40, 2))
    threesp = pd.Series(np.full(40, 3))

    temp_user_df['cat2'] = (pd.concat([onesp, twosp, threesp], axis=0, ignore_index=True))

    sub_cat = pd.Series(np.arange(1, 41))
    temp_user_df['cat3'] = pd.concat([sub_cat] * 3, axis=0, ignore_index=True)
    temp_user_df['numeric1'] = (np.random.random_sample((120,))) * 10
    temp_user_df['target'] = np.random.random_sample((120,))

    user_df = user_df.append(temp_user_df, ignore_index=True)

user_df.to_csv('user_df.csv', index = False)  # write to csv, optional



'''
Synthetic data to simulate live model refit
'''

user_id = num_users -5  # user_id used for prediction, can be any int < num_users

#Initialize df with one user
live_df = pd.DataFrame(pd.Series(np.full(120, user_id)), columns = ['user_id'])

# category 1
live_df['cat1'] = pd.Series(np.arange(1,121))

# category 2
onesp = pd.Series(np.full(40, 1))
twosp = pd.Series(np.full(40, 2))
threesp = pd.Series(np.full(40, 3))
live_df['cat2'] = (pd.concat([onesp, twosp, threesp], axis = 0, ignore_index=True))

# category 3
sub_cat = pd.Series(np.arange(1,41))
live_df['cat3'] = pd.concat([sub_cat]*3, axis = 0, ignore_index=True)

# numeric feature
live_df['numeric1'] = (np.random.random_sample((120,))) * 10

#target
live_df['target'] = np.random.random_sample((120,))

live_df.to_csv('live_df.csv', index = False)  # write to csv, optional



'''
Create a synthetic test df
'''

#user_id = num_users -5    # user_id used for prediction, can be any int < num_users
df_len = 200    # Length of test df

test_df = pd.DataFrame(pd.Series(np.full(df_len, user_id)), columns = ['user_id'])
test_df['cat1'] = np.random.randint(1, 121, size=(df_len, 1))
test_df['cat2'] = np.random.randint(1, 4, size=(df_len, 1))
test_df['cat3'] = np.random.randint(1, 41, size=(df_len, 1))
test_df['numeric1'] = (np.random.random_sample((df_len,))) * 10

test_df.to_csv('prediction_df.csv', index = False)  # write to csv

'''
Data for cosine similarity recommender data
'''
num_cats = 120
recommender_user_df = pd.DataFrame()
temp_recommender_user_df = pd.DataFrame()
temp_recommender_user_df['user_id'] = 1

for i in range(1, (num_users + 1)):
    temp_recommender_user_df['user_id'] = i
    # print(i, temp_recommender_user_df['user_id'].values)
    for j in range(1, (num_cats + 1)):
        temp_recommender_user_df['cat_{}'.format(j)] = np.random.random_sample((1,))
    recommender_user_df = recommender_user_df.append(temp_recommender_user_df, ignore_index=True)
    recommender_user_df = recommender_user_df.fillna(1)
    recommender_user_df['user_id'] = recommender_user_df['user_id'].astype(int)

recommender_user_df.to_csv('recommender_df.csv', index=False)

