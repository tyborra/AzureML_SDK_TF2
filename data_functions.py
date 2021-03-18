import pandas as pd
import numpy as np

def standard_scaler(df, cols):
    '''
    scales columns inplace, returns dict of std and mean for portability
    to use dict on new data use (df[col] - scaler_dict[col][0]) /  scaler_dict[col][1])
    '''
    scaler_dict = {}
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        scaler_dict[col] = mean, std
        df[col] = (df[col] - mean) / std
    return scaler_dict


def label_encoder(df, cols):
    '''
    encodes columns inplace, returns dict of categorical embeddings
    to use dict on new data use: df.replace(transform_dict)
    '''
    transform_dict = {}
    for col in cols:
        cats = pd.Categorical(df[col]).categories
        d = {}
        for i, cat in enumerate(cats):
            d[cat] = i
        transform_dict[col] = d
    df.replace(transform_dict)
    return transform_dict


def ingest_data(df):
    '''
    Ingest training data for TF.
    input: raw train df
    returns: modified df and target as categorical y
    '''
    df.fillna(0)
    df = ewma_to_cat(df)
    y = df['target']
    df = df.drop(['target'], axis=1)
    return df, y


def ewma_to_cat(df):
    '''
    Convert target to discrete based on IQR
    returns: modified df
    '''
    q1, q2, q3 = np.percentile(df['target'], [25, 50, 75])
    df.loc[df['target'] > q2, 'target'] = 1
    df.loc[df['target'] <= q2, 'target'] = 0
    # user_df.loc[(user_df['target'] >= q1) & (user_df['target'] <= q3), 'target'] = 1    #group Q2 and Q3 to make trinary
    df['target'] = df['target'].astype(int)
    return df


def prepare_data(df, cat_features, num_features):
    '''
    prepares categorical features for tf inplace
    input: df and lists of categorical features and numeric features
    output: scaling and label encoding dicts, for later use
    '''

    # label encoder
    label_dict = label_encoder(df, cat_features)

    # scaler
    scaler_dict = standard_scaler(df, num_features)

    return label_dict, scaler_dict